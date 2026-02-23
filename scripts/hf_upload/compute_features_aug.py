#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "scipy",
#   "librosa",
#   "soundfile",
#   "pretty-midi",
#   "pyarrow",
#   "huggingface-hub",
# ]
# ///
"""Compute augmented mel spectrograms and onset labels for ADT datasets.

Standalone PEP 723 UV script for HF Jobs. Downloads raw audio datasets from
HF Hub, applies waveform augmentations (RIR convolution, EQ, noise mixing),
computes mel spectrograms + onset targets, and streams sharded Parquet files
to HF Hub.

Processes TRAIN split only -- val/test stay unaugmented for clean evaluation.
For each track, produces 1 dry (unaugmented) copy + N augmented copies.

Usage (HF Jobs):
    hf jobs uv run scripts/hf_upload/compute_features_aug.py \\
        --flavor a10g-large --timeout 48h \\
        --secret HF_TOKEN=$HF_TOKEN \\
        -- --dataset egmd --augment-copies 3 --output-repo schismaudio/e-gmd

Local testing:
    uv run scripts/hf_upload/compute_features_aug.py \\
        --dataset egmd --max-entries 10 --no-upload --output-dir /tmp/aug_out
"""

import argparse
import csv
import json
import os
import resource
import sys
from pathlib import Path

# Force line-buffered stdout for real-time HF Jobs log visibility
sys.stdout.reconfigure(line_buffering=True)

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, sosfilt


# ---------------------------------------------------------------------------
# Audio & target parameters (must match drumscribble.config / audio / targets)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
HOP_LENGTH = 256       # 16000/256 = 62.5 fps
N_FFT = 2048
N_MELS = 128
F_MIN = 20.0
F_MAX = 8000.0
FPS = SAMPLE_RATE / HOP_LENGTH  # 62.5

NUM_CLASSES = 26
TARGET_WIDENING = [0.3, 0.6, 1.0, 0.6, 0.3]

GM_CLASSES = sorted([
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59,
    75, 77,
])
GM_NOTE_TO_INDEX = {note: i for i, note in enumerate(GM_CLASSES)}

# Roland TD-17 non-standard MIDI notes -> GM standard (E-GMD specific)
EGMD_NOTE_REMAP = {
    22: 42,   # HH closed edge -> Closed Hi-Hat
    26: 46,   # HH open edge -> Open Hi-Hat
    58: 43,   # Tom3 rim -> High Floor Tom
}

# STAR 18-class abbreviations -> GM note numbers
STAR_ABBREV_TO_GM = {
    "BD": 36, "SD": 38, "CHH": 42, "PHH": 44, "OHH": 46,
    "HT": 48, "MT": 45, "LT": 43, "CRC": 49, "SPC": 55,
    "CHC": 52, "RD": 51, "RB": 53, "CB": 56, "CL": 75,
    "CLP": 39, "SS": 37, "TB": 54,
}


# ---------------------------------------------------------------------------
# Audio augmentation (inlined from audio_augment.py for HF Jobs standalone)
# ---------------------------------------------------------------------------

def apply_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    wet_mix: float = 0.7,
) -> np.ndarray:
    """Convolve audio with a room impulse response.

    Args:
        audio: 1-D float32 audio signal.
        rir: 1-D float32 impulse response.
        wet_mix: Blend factor (0.0 = dry, 1.0 = fully wet).

    Returns:
        Blended audio, same length as input.
    """
    wet = fftconvolve(audio, rir, mode="full")[: len(audio)]
    # Normalize wet to match dry RMS to prevent volume jumps
    dry_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-8
    wet = wet * (dry_rms / wet_rms)
    return ((1.0 - wet_mix) * audio + wet_mix * wet).astype(np.float32)


def _low_shelf_sos(freq: float, gain_db: float, sr: int) -> np.ndarray:
    """Design a low shelf biquad filter as second-order sections."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * 0.707)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _high_shelf_sos(freq: float, gain_db: float, sr: int) -> np.ndarray:
    """Design a high shelf biquad filter as second-order sections."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * 0.707)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _peak_sos(freq: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """Design a peak/notch biquad filter as second-order sections."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def apply_eq(
    audio: np.ndarray,
    sr: int,
    low_shelf_db: float = 0.0,
    high_shelf_db: float = 0.0,
    mid_freq: float = 1000.0,
    mid_db: float = 0.0,
    mid_q: float = 1.0,
    low_shelf_freq: float = 150.0,
    high_shelf_freq: float = 6000.0,
) -> np.ndarray:
    """Apply parametric EQ (low shelf + mid peak + high shelf).

    All gains in dB. Zero gain = passthrough.
    """
    sos_list = []
    if abs(low_shelf_db) > 0.01:
        sos_list.append(_low_shelf_sos(low_shelf_freq, low_shelf_db, sr))
    if abs(mid_db) > 0.01:
        sos_list.append(_peak_sos(mid_freq, mid_db, mid_q, sr))
    if abs(high_shelf_db) > 0.01:
        sos_list.append(_high_shelf_sos(high_shelf_freq, high_shelf_db, sr))
    if not sos_list:
        return audio.copy()
    sos = np.concatenate(sos_list, axis=0)
    return sosfilt(sos, audio).astype(np.float32)


def apply_noise(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 30.0,
) -> np.ndarray:
    """Add noise at a target SNR.

    Noise is looped if shorter than audio, trimmed if longer.
    """
    if len(noise) < len(audio):
        repeats = (len(audio) // len(noise)) + 1
        noise = np.tile(noise, repeats)
    noise = noise[: len(audio)]
    sig_power = np.mean(audio ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return (audio + scale * noise).astype(np.float32)


def augment_audio(
    audio: np.ndarray,
    sr: int,
    rirs: list[np.ndarray],
    noises: list[np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Generate one augmented variant of an audio signal.

    Returns:
        Tuple of (augmented_audio, augmentation_params_dict).
    """
    aug = audio.copy()

    rir_idx = int(rng.integers(len(rirs)))
    rir = rirs[rir_idx]
    wet_mix = float(rng.uniform(0.3, 0.9))
    aug = apply_rir(aug, rir, wet_mix=wet_mix)

    low_shelf_db = float(rng.uniform(-6.0, 6.0))
    high_shelf_db = float(rng.uniform(-6.0, 6.0))
    low_shelf_freq = float(rng.uniform(80.0, 200.0))
    high_shelf_freq = float(rng.uniform(4000.0, 8000.0))
    mid_freq = float(rng.uniform(300.0, 3000.0))
    mid_db = float(rng.uniform(-4.0, 4.0))
    mid_q = float(rng.uniform(0.7, 2.0))
    aug = apply_eq(
        aug, sr=sr,
        low_shelf_db=low_shelf_db,
        high_shelf_db=high_shelf_db,
        low_shelf_freq=low_shelf_freq,
        high_shelf_freq=high_shelf_freq,
        mid_freq=mid_freq,
        mid_db=mid_db,
        mid_q=mid_q,
    )

    noise_idx = int(rng.integers(len(noises)))
    noise = noises[noise_idx]
    snr_db = float(rng.uniform(20.0, 40.0))
    aug = apply_noise(aug, noise, snr_db=snr_db)

    params = {
        "rir_idx": rir_idx,
        "wet_mix": round(wet_mix, 3),
        "low_shelf_db": round(low_shelf_db, 2),
        "high_shelf_db": round(high_shelf_db, 2),
        "low_shelf_freq": round(low_shelf_freq, 1),
        "high_shelf_freq": round(high_shelf_freq, 1),
        "mid_freq": round(mid_freq, 1),
        "mid_db": round(mid_db, 2),
        "mid_q": round(mid_q, 2),
        "noise_idx": noise_idx,
        "snr_db": round(snr_db, 1),
    }
    return aug, params


# ---------------------------------------------------------------------------
# Feature computation (inlined from compute_features.py for HF Jobs standalone)
# ---------------------------------------------------------------------------

def load_audio_mono(path: str) -> np.ndarray:
    """Load audio file, convert to mono, resample to SAMPLE_RATE.

    Returns: 1-D numpy array at SAMPLE_RATE.
    """
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def compute_mel(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram. Returns (n_mels, n_frames) float32."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0,
    )
    return np.log(np.maximum(mel, 1e-7)).astype(np.float32)


def compute_onset_targets(
    events: list[tuple[float, int, int]],
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert onset events to frame-level targets with widening.

    Args:
        events: List of (time_seconds, gm_note, velocity).
        n_frames: Number of spectrogram frames.

    Returns:
        onset_targets: (n_classes, n_frames) float32 array [0, 1].
        velocity_targets: (n_classes, n_frames) float32 array [0, 1].
    """
    onset = np.zeros((NUM_CLASSES, n_frames), dtype=np.float32)
    vel = np.zeros((NUM_CLASSES, n_frames), dtype=np.float32)
    half_w = len(TARGET_WIDENING) // 2

    for time_s, gm_note, velocity in events:
        # Remap non-standard notes
        gm_note = EGMD_NOTE_REMAP.get(gm_note, gm_note)
        if gm_note not in GM_NOTE_TO_INDEX:
            continue
        cls_idx = GM_NOTE_TO_INDEX[gm_note]
        center = round(time_s * FPS)
        vel_norm = velocity / 127.0

        for i, w in enumerate(TARGET_WIDENING):
            frame = center - half_w + i
            if 0 <= frame < n_frames:
                onset[cls_idx, frame] = max(onset[cls_idx, frame], w)
                if w == 1.0:
                    vel[cls_idx, frame] = vel_norm

    return onset, vel


# ---------------------------------------------------------------------------
# Annotation parsers
# ---------------------------------------------------------------------------

def parse_midi_events(midi_path: str) -> list[tuple[float, int, int]]:
    """Parse MIDI file to (time_seconds, gm_note, velocity) events."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                events.append((note.start, note.pitch, note.velocity))
    events.sort(key=lambda x: x[0])
    return events


def parse_star_annotation(ann_path: str) -> list[tuple[float, int, int]]:
    """Parse STAR TSV annotation to (time_seconds, gm_note, velocity) events."""
    events = []
    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            time_s = float(parts[0])
            abbrev = parts[1]
            velocity = int(parts[2])
            gm_note = STAR_ABBREV_TO_GM.get(abbrev)
            if gm_note is not None:
                events.append((time_s, gm_note, velocity))
    events.sort(key=lambda x: x[0])
    return events


# ---------------------------------------------------------------------------
# Dataset-specific entry discovery
# ---------------------------------------------------------------------------

def discover_egmd_entries(root: Path) -> list[dict]:
    """Discover E-GMD or GMD entries from CSV manifest."""
    csv_candidates = [
        root / "e-gmd-v1.0.0.csv",
        root / "info.csv",          # GMD uses info.csv
    ]
    csv_path = None
    for c in csv_candidates:
        if c.exists():
            csv_path = c
            break
    if csv_path is None:
        raise FileNotFoundError(f"No CSV manifest found in {root}")

    entries = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_path = root / row["audio_filename"]
            midi_path = root / row["midi_filename"]
            if not audio_path.exists() or not midi_path.exists():
                continue
            entries.append({
                "audio_path": str(audio_path),
                "midi_path": str(midi_path),
                "split": row.get("split", "train"),
                "duration": float(row.get("duration", 0)),
                "style": row.get("style", ""),
                "bpm": int(row["bpm"]) if row.get("bpm") else 0,
                "drummer": row.get("drummer", ""),
                "session": row.get("session", ""),
                "beat_type": row.get("beat_type", ""),
                "time_signature": row.get("time_signature", ""),
                "kit_name": row.get("kit_name", ""),
                "source_id": row.get("id", ""),
            })
    return entries


def discover_star_entries(root: Path) -> list[dict]:
    """Discover STAR Drums entries from directory structure."""
    entries = []
    for split_name in ["training", "validation", "test"]:
        split_dir = root / "data" / split_name
        if not split_dir.exists():
            continue
        for source_dir in sorted(split_dir.iterdir()):
            if not source_dir.is_dir():
                continue
            ann_dir = source_dir / "annotation"
            audio_dir = source_dir / "audio" / "mix"
            if not ann_dir.exists() or not audio_dir.exists():
                continue
            for ann_file in sorted(ann_dir.glob("*.txt")):
                audio_file = audio_dir / ann_file.stem
                # Try common extensions
                for ext in [".flac", ".wav", ".mp3"]:
                    candidate = audio_dir / (ann_file.stem + ext)
                    if candidate.exists():
                        audio_file = candidate
                        break
                else:
                    continue

                entries.append({
                    "audio_path": str(audio_file),
                    "annotation_path": str(ann_file),
                    "split": split_name if split_name != "training" else "train",
                    "source": source_dir.name,
                    "track_id": ann_file.stem,
                })
    return entries


# ---------------------------------------------------------------------------
# Feature row builders
# ---------------------------------------------------------------------------

def build_feature_row(
    audio: np.ndarray,
    events: list[tuple[float, int, int]],
    entry: dict,
    dataset: str,
    augmentation: str,
) -> dict:
    """Build a single feature row dict from audio and annotations.

    Args:
        audio: 1-D float32 audio at SAMPLE_RATE.
        events: Onset events [(time_s, gm_note, velocity), ...].
        entry: Original entry dict with metadata.
        dataset: 'egmd' or 'star'.
        augmentation: Empty string for dry, JSON string for augmented.

    Returns:
        Dict suitable for Parquet row.
    """
    mel = compute_mel(audio)
    n_frames = mel.shape[1]
    onset, vel = compute_onset_targets(events, n_frames)

    row = {
        "mel_spectrogram": mel.tobytes(),
        "onset_targets": onset.tobytes(),
        "velocity_targets": vel.tobytes(),
        "n_frames": n_frames,
        "n_mels": N_MELS,
        "n_classes": NUM_CLASSES,
        "sample_rate": SAMPLE_RATE,
        "hop_length": HOP_LENGTH,
        "fps": FPS,
        "duration": len(audio) / SAMPLE_RATE,
        "split": entry["split"],
        "augmentation": augmentation,
        "source_audio": Path(entry["audio_path"]).name,
    }

    if dataset == "egmd":
        row.update({
            "style": entry.get("style", ""),
            "bpm": entry.get("bpm", 0),
            "drummer": entry.get("drummer", ""),
            "session": entry.get("session", ""),
            "beat_type": entry.get("beat_type", ""),
            "time_signature": entry.get("time_signature", ""),
            "kit_name": entry.get("kit_name", ""),
            "source_id": entry.get("source_id", ""),
        })
    elif dataset == "star":
        row.update({
            "source": entry.get("source", ""),
            "track_id": entry.get("track_id", ""),
        })

    return row


# ---------------------------------------------------------------------------
# Parquet shard writing and uploading
# ---------------------------------------------------------------------------

def write_and_upload_shard(
    rows: list[dict],
    shard_idx: int,
    output_dir: Path,
    output_repo: str | None,
    no_upload: bool,
) -> str:
    """Write a Parquet shard and optionally upload to HF Hub.

    Args:
        rows: List of feature row dicts.
        shard_idx: Shard index number.
        output_dir: Local output directory.
        output_repo: HF Hub repo ID (e.g. 'schismaudio/e-gmd').
        no_upload: If True, keep local file and skip upload.

    Returns:
        Shard filename.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"train-aug-{shard_idx:05d}.parquet"
    filepath = output_dir / filename

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, filepath, compression="zstd")
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  Wrote {filename} ({len(rows)} rows, {size_mb:.1f} MB)")

    if not no_upload and output_repo:
        from huggingface_hub import HfApi
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        uploaded = False
        try:
            api.upload_file(
                path_or_fileobj=str(filepath),
                path_in_repo=f"features/{filename}",
                repo_id=output_repo,
                repo_type="dataset",
            )
            uploaded = True
            print(f"  Uploaded {filename} to {output_repo}")
        except Exception as e:
            print(f"  WARNING: Upload failed for {filename}: {e}")
            print(f"  Keeping local file for retry.")
        if uploaded:
            os.unlink(filepath)
            print(f"  Deleted local {filename}")

    return filename


# ---------------------------------------------------------------------------
# RIR and noise loading
# ---------------------------------------------------------------------------

def load_wav_files(wav_paths: list[Path]) -> list[np.ndarray]:
    """Load WAV files as mono float32 arrays at SAMPLE_RATE."""
    arrays = []
    for p in wav_paths:
        try:
            audio, sr = sf.read(str(p), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            arrays.append(audio)
        except Exception as e:
            print(f"  SKIP loading {p.name}: {e}", file=sys.stderr)
    return arrays


def log_rss(prefix: str = "") -> float:
    """Log current RSS memory usage and return value in MB.

    On Linux, reads /proc/self/status for current (not peak) RSS.
    On macOS, falls back to peak RSS via resource module.
    """
    rss_mb = 0.0
    if sys.platform == "linux":
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = int(line.split()[1]) / 1024  # KB -> MB
                        break
        except OSError:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    elif sys.platform == "darwin":
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    else:
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"{prefix}RSS: {rss_mb:.0f} MB")
    return rss_mb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute augmented mel spectrograms and onset labels for ADT datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["egmd", "star"],
        required=True,
        help="Dataset type to process.",
    )
    parser.add_argument(
        "--augment-copies",
        type=int,
        default=3,
        help="Number of augmented variants per track (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Process only N entries (for local testing).",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip HF Hub upload; write to --output-dir instead.",
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        default=None,
        help="HF Hub dataset repo to upload shards to.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/features_aug"),
        help="Local output directory for Parquet shards (default: /tmp/features_aug).",
    )
    parser.add_argument(
        "--max-shard-mb",
        type=int,
        default=500,
        help="Max shard size in MB before flushing (default: 500).",
    )
    args = parser.parse_args()

    if not args.no_upload and not args.output_repo:
        print("ERROR: --output-repo is required unless --no-upload is set.", file=sys.stderr)
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    max_shard_bytes = args.max_shard_mb * 1024 * 1024

    print("=" * 60)
    print("Augmented Feature Computation")
    print("=" * 60)
    print(f"Dataset:         {args.dataset}")
    print(f"Augment copies:  {args.augment_copies}")
    print(f"Seed:            {args.seed}")
    print(f"Max entries:     {args.max_entries or 'all'}")
    print(f"Upload:          {'disabled' if args.no_upload else args.output_repo}")
    print(f"Output dir:      {args.output_dir}")
    print(f"Max shard size:  {args.max_shard_mb} MB")
    print(f"Params:          sr={SAMPLE_RATE}, hop={HOP_LENGTH}, n_mels={N_MELS}, "
          f"fps={FPS}, classes={NUM_CLASSES}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Download raw dataset from HF Hub
    # ------------------------------------------------------------------
    from huggingface_hub import snapshot_download

    token = os.environ.get("HF_TOKEN")

    if args.dataset == "egmd":
        print("Downloading E-GMD dataset...")
        dataset_dir = Path(snapshot_download(
            "schismaudio/e-gmd", repo_type="dataset", token=token,
        ))
        print(f"  E-GMD at: {dataset_dir}")
    elif args.dataset == "star":
        print("Downloading STAR Drums dataset...")
        dataset_dir = Path(snapshot_download(
            "schismaudio/star-drums", repo_type="dataset", token=token,
        ))
        print(f"  STAR at: {dataset_dir}")

    log_rss("After dataset download | ")

    # ------------------------------------------------------------------
    # Step 2: Download OpenSLR RIRs
    # ------------------------------------------------------------------
    print("\nDownloading OpenSLR RIRs...")
    rir_dir = Path(snapshot_download(
        "schismaudio/openslr-rirs", repo_type="dataset", token=token,
    ))
    print(f"  RIRs at: {rir_dir}")
    log_rss("After RIR download | ")

    # ------------------------------------------------------------------
    # Step 3: Load RIR and noise WAV files
    # ------------------------------------------------------------------
    print("\nLoading RIR files...")
    rir_paths = sorted(
        list((rir_dir / "RIRS_NOISES" / "real_rirs_isotropic_noises").rglob("*.wav"))
        + list((rir_dir / "RIRS_NOISES" / "simulated_rirs").rglob("*.wav"))
    )
    print(f"  Found {len(rir_paths)} RIR files")
    rirs = load_wav_files(rir_paths)
    print(f"  Loaded {len(rirs)} RIRs")
    log_rss("After RIR load | ")

    print("\nLoading noise files...")
    noise_paths = sorted(
        list((rir_dir / "RIRS_NOISES" / "pointsource_noises").rglob("*.wav"))
    )
    print(f"  Found {len(noise_paths)} noise files")
    noises = load_wav_files(noise_paths)
    print(f"  Loaded {len(noises)} noises")
    log_rss("After noise load | ")

    if not rirs:
        print("ERROR: No RIRs loaded. Cannot proceed.", file=sys.stderr)
        sys.exit(1)
    if not noises:
        print("ERROR: No noises loaded. Cannot proceed.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 4: Discover entries and filter to train split
    # ------------------------------------------------------------------
    print(f"\nDiscovering {args.dataset} entries...")
    if args.dataset == "egmd":
        all_entries = discover_egmd_entries(dataset_dir)
    elif args.dataset == "star":
        all_entries = discover_star_entries(dataset_dir)

    print(f"  Total entries: {len(all_entries)}")

    # Filter to train split only
    entries = [e for e in all_entries if e["split"] == "train"]
    print(f"  Train entries: {len(entries)}")

    if args.max_entries is not None:
        entries = entries[: args.max_entries]
        print(f"  Limited to: {len(entries)} entries")

    if not entries:
        print("ERROR: No train entries found.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 5: Process entries with augmentation
    # ------------------------------------------------------------------
    print(f"\nProcessing {len(entries)} entries "
          f"(1 dry + {args.augment_copies} augmented = "
          f"{1 + args.augment_copies} copies each)...")
    print(f"Expected total rows: {len(entries) * (1 + args.augment_copies)}")
    print()

    rows: list[dict] = []
    current_bytes = 0
    shard_idx = 0
    total_rows = 0
    total_skipped = 0

    for entry_idx, entry in enumerate(entries):
        try:
            # Load audio
            audio = load_audio_mono(entry["audio_path"])

            # Parse annotations
            if args.dataset == "egmd":
                events = parse_midi_events(entry["midi_path"])
            elif args.dataset == "star":
                events = parse_star_annotation(entry["annotation_path"])

            # (a) Dry copy -- unaugmented
            row = build_feature_row(audio, events, entry, args.dataset, augmentation="")
            row_bytes = len(row["mel_spectrogram"]) + len(row["onset_targets"]) + len(row["velocity_targets"])
            rows.append(row)
            current_bytes += row_bytes

            # (b) Augmented copies
            for copy_idx in range(args.augment_copies):
                aug_audio, aug_params = augment_audio(
                    audio, sr=SAMPLE_RATE, rirs=rirs, noises=noises, rng=rng,
                )
                aug_params["copy"] = copy_idx
                aug_json = json.dumps(aug_params)
                aug_row = build_feature_row(
                    aug_audio, events, entry, args.dataset, augmentation=aug_json,
                )
                aug_row_bytes = (
                    len(aug_row["mel_spectrogram"])
                    + len(aug_row["onset_targets"])
                    + len(aug_row["velocity_targets"])
                )
                rows.append(aug_row)
                current_bytes += aug_row_bytes

            # Check if shard should be flushed
            if current_bytes >= max_shard_bytes:
                write_and_upload_shard(
                    rows, shard_idx, args.output_dir, args.output_repo, args.no_upload,
                )
                total_rows += len(rows)
                shard_idx += 1
                rows = []
                current_bytes = 0

        except Exception as e:
            print(f"  SKIP entry {entry_idx} ({entry.get('audio_path', '?')}): {e}",
                  file=sys.stderr)
            total_skipped += 1

        # Progress log every 100 entries
        if (entry_idx + 1) % 100 == 0:
            log_rss(
                f"  [{entry_idx + 1}/{len(entries)}] "
                f"rows_buffered={len(rows)} shards_written={shard_idx} | "
            )

    # ------------------------------------------------------------------
    # Step 6: Flush final shard
    # ------------------------------------------------------------------
    if rows:
        write_and_upload_shard(
            rows, shard_idx, args.output_dir, args.output_repo, args.no_upload,
        )
        total_rows += len(rows)
        shard_idx += 1

    # ------------------------------------------------------------------
    # Step 7: Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Dataset:         {args.dataset}")
    print(f"Entries:         {len(entries)} (train split)")
    print(f"Skipped:         {total_skipped}")
    print(f"Total rows:      {total_rows}")
    print(f"Shards written:  {shard_idx}")
    print(f"Augment copies:  {args.augment_copies}")
    if not args.no_upload:
        print(f"Uploaded to:     {args.output_repo}")
    else:
        print(f"Output dir:      {args.output_dir}")
    log_rss("Final | ")
    print("Done.")


if __name__ == "__main__":
    main()
