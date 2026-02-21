#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub[hf_xet]",
#     "numpy",
#     "soundfile",
#     "librosa",
#     "pretty_midi",
#     "pyarrow",
#     "datasets",
# ]
# ///
"""HF Jobs script: compute mel spectrograms + onset/velocity targets.

Downloads a raw drum dataset from HF Hub, computes pre-processed features
(mel spectrograms, onset targets, velocity targets), and uploads them back
as a ``features/`` config in the same dataset repo.

Designed to run on HF Jobs infrastructure (CPU instance, ~2-3h).

Usage (via ``uv run`` or HF Jobs):
    HF_TOKEN=hf_... python compute_features_job.py --dataset egmd --repo schismaudio/e-gmd
    HF_TOKEN=hf_... python compute_features_job.py --dataset star --repo schismaudio/star-drums
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import HfApi, snapshot_download

# ---------------------------------------------------------------------------
# Audio & target parameters (must match drumscribble.config)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
HOP_LENGTH = 256  # 16000 / 256 = 62.5 fps
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
    22: 42,  # HH closed edge -> Closed Hi-Hat
    26: 46,  # HH open edge -> Open Hi-Hat
    58: 43,  # Tom3 rim -> High Floor Tom
}

# STAR 18-class abbreviations -> GM note numbers
STAR_ABBREV_TO_GM = {
    "BD": 36, "SD": 38, "CHH": 42, "PHH": 44, "OHH": 46,
    "HT": 48, "MT": 45, "LT": 43, "CRC": 49, "SPC": 55,
    "CHC": 52, "RD": 51, "RB": 53, "CB": 56, "CL": 75,
    "CLP": 39, "SS": 37, "TB": 54,
}


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def load_audio_mono(path: str) -> np.ndarray:
    """Load audio file, convert to mono, resample to SAMPLE_RATE."""
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
    """Discover E-GMD entries from CSV manifest."""
    csv_candidates = [
        root / "e-gmd-v1.0.0.csv",
        root / "info.csv",
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
                    "split": "train" if split_name == "training" else split_name,
                    "source": source_dir.name,
                    "track_id": ann_file.stem,
                })
    return entries


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

def process_egmd_entry(entry: dict) -> dict | None:
    """Process one E-GMD entry into a feature row."""
    try:
        audio = load_audio_mono(entry["audio_path"])
        mel = compute_mel(audio)
        n_frames = mel.shape[1]
        events = parse_midi_events(entry["midi_path"])
        onset, vel = compute_onset_targets(events, n_frames)

        return {
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
            "style": entry.get("style", ""),
            "bpm": entry.get("bpm", 0),
            "drummer": entry.get("drummer", ""),
            "session": entry.get("session", ""),
            "beat_type": entry.get("beat_type", ""),
            "time_signature": entry.get("time_signature", ""),
            "kit_name": entry.get("kit_name", ""),
            "source_id": entry.get("source_id", ""),
            "source_audio": Path(entry["audio_path"]).name,
        }
    except Exception as e:
        print(f"  SKIP {entry['audio_path']}: {e}", file=sys.stderr)
        return None


def process_star_entry(entry: dict) -> dict | None:
    """Process one STAR Drums entry into a feature row."""
    try:
        audio = load_audio_mono(entry["audio_path"])
        mel = compute_mel(audio)
        n_frames = mel.shape[1]
        events = parse_star_annotation(entry["annotation_path"])
        onset, vel = compute_onset_targets(events, n_frames)

        return {
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
            "source": entry.get("source", ""),
            "track_id": entry.get("track_id", ""),
            "source_audio": Path(entry["audio_path"]).name,
        }
    except Exception as e:
        print(f"  SKIP {entry['audio_path']}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Parquet writing
# ---------------------------------------------------------------------------

def write_parquet_shards(
    rows: list[dict],
    output_dir: Path,
    split: str,
    max_shard_bytes: int = 500 * 1024 * 1024,
) -> int:
    """Write feature rows to sharded zstd-compressed parquet files.

    Output: output_dir/{split}-00000-of-NNNNN.parquet

    Returns the number of shards written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"  No rows for split '{split}', skipping.")
        return 0

    # Estimate shard boundaries by feature tensor sizes
    shards: list[list[dict]] = []
    current_shard: list[dict] = []
    current_bytes = 0
    for row in rows:
        row_bytes = (
            len(row["mel_spectrogram"])
            + len(row["onset_targets"])
            + len(row["velocity_targets"])
        )
        if current_bytes + row_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_bytes = 0
        current_shard.append(row)
        current_bytes += row_bytes
    if current_shard:
        shards.append(current_shard)

    n_shards = len(shards)
    for shard_idx, shard_rows in enumerate(shards):
        filename = f"{split}-{shard_idx:05d}-of-{n_shards:05d}.parquet"
        filepath = output_dir / filename

        table = pa.Table.from_pylist(shard_rows)
        pq.write_table(table, filepath, compression="zstd")
        print(f"  Wrote {filepath.name} ({len(shard_rows)} rows)")

    return n_shards


# ---------------------------------------------------------------------------
# HF Hub integration
# ---------------------------------------------------------------------------

def download_dataset(repo: str, token: str) -> Path:
    """Download the raw dataset from HF Hub to a local cache directory."""
    print(f"Downloading {repo} from HF Hub...")
    t0 = time.monotonic()
    local_dir = Path(snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        token=token,
        local_dir=f"/tmp/{repo.replace('/', '_')}_raw",
    ))
    elapsed = time.monotonic() - t0
    print(f"Downloaded to {local_dir} ({elapsed:.1f}s)")
    return local_dir


def upload_features(
    repo: str,
    features_dir: Path,
    token: str,
) -> None:
    """Upload the features/ directory to HF Hub as a config."""
    api = HfApi(token=token)
    print(f"Uploading features to {repo} (features/ config)...")
    t0 = time.monotonic()
    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(features_dir),
        path_in_repo="features/",
        commit_message="Add pre-computed mel spectrogram + onset/velocity features",
    )
    elapsed = time.monotonic() - t0
    print(f"Upload complete ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HF Jobs script: compute dataset features and upload to Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["egmd", "star"],
        help="Dataset type to process.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HF dataset repo ID (e.g. schismaudio/e-gmd).",
    )
    parser.add_argument(
        "--max-shard-mb",
        type=int,
        default=500,
        help="Max parquet shard size in MB (default: 500).",
    )
    args = parser.parse_args()

    # Read token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN environment variable is required.", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset:  {args.dataset}")
    print(f"Repo:     {args.repo}")
    print(f"Shard MB: {args.max_shard_mb}")
    print(f"Params:   sr={SAMPLE_RATE}, hop={HOP_LENGTH}, n_mels={N_MELS}, "
          f"fps={FPS}, classes={NUM_CLASSES}")
    print()

    # Step 1: Download raw dataset from HF Hub
    raw_dir = download_dataset(args.repo, token)
    print()

    # Step 2: Discover entries
    if args.dataset == "egmd":
        entries = discover_egmd_entries(raw_dir)
        process_fn = process_egmd_entry
    elif args.dataset == "star":
        entries = discover_star_entries(raw_dir)
        process_fn = process_star_entry
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Found {len(entries)} entries")
    if not entries:
        print("No entries found. Check that the dataset downloaded correctly.",
              file=sys.stderr)
        sys.exit(1)

    # Group by split
    splits: dict[str, list[dict]] = {}
    for entry in entries:
        s = entry["split"]
        splits.setdefault(s, []).append(entry)

    for s, s_entries in sorted(splits.items()):
        print(f"  {s}: {len(s_entries)} entries")
    print()

    # Step 3: Process each split and write parquet shards
    features_dir = Path(f"/tmp/{args.repo.replace('/', '_')}_features")
    max_shard_bytes = args.max_shard_mb * 1024 * 1024
    total_processed = 0
    total_skipped = 0

    t0 = time.monotonic()
    for split_name, split_entries in sorted(splits.items()):
        print(f"=== Processing split: {split_name} ({len(split_entries)} entries) ===")
        rows: list[dict] = []
        for i, entry in enumerate(split_entries):
            row = process_fn(entry)
            if row is not None:
                rows.append(row)
            else:
                total_skipped += 1

            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(split_entries)} processed ({len(rows)} ok)")

        print(f"  Completed: {len(rows)} features, {total_skipped} skipped")
        write_parquet_shards(rows, features_dir, split_name, max_shard_bytes)
        total_processed += len(rows)
        print()

    elapsed = time.monotonic() - t0
    print(f"Feature computation done: {total_processed} rows, "
          f"{total_skipped} skipped ({elapsed:.1f}s)")
    print()

    # Step 4: Upload features to HF Hub
    upload_features(args.repo, features_dir, token)

    print()
    print(f"All done. {total_processed} feature rows uploaded to "
          f"{args.repo} (features/ config).")


if __name__ == "__main__":
    main()
