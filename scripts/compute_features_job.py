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

Streams a raw drum dataset from HF Hub (no full download), computes
pre-processed features (mel spectrograms, onset targets, velocity targets),
and uploads them back as a ``features/`` config in the same dataset repo.

Designed to run on HF Jobs infrastructure (CPU instance, ~2-3h).

Usage (via ``uv run`` or HF Jobs):
    HF_TOKEN=hf_... python compute_features_job.py --dataset egmd --repo schismaudio/e-gmd
    HF_TOKEN=hf_... python compute_features_job.py --dataset star --repo zkeown/star-drums
"""

from __future__ import annotations

import argparse
import io
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
from huggingface_hub import HfApi, hf_hub_download

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

def load_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode audio from in-memory bytes, convert to mono, resample."""
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def load_audio_mono(path: str) -> np.ndarray:
    """Load audio file from disk, convert to mono, resample to SAMPLE_RATE."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def parse_midi_from_bytes(midi_bytes: bytes) -> list[tuple[float, int, int]]:
    """Parse MIDI from in-memory bytes to (time, note, velocity) events."""
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            for note in inst.notes:
                events.append((note.start, note.pitch, note.velocity))
    events.sort(key=lambda x: x[0])
    return events


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
# Parquet shard writing
# ---------------------------------------------------------------------------

def _write_and_upload_shard(
    rows: list[dict],
    split: str,
    shard_idx: int,
    repo: str,
    token: str,
) -> None:
    """Write a Parquet shard, upload it to HF Hub, then delete local file."""
    tmp_path = Path(f"/tmp/_shard_{split}_{shard_idx:05d}.parquet")
    filename = f"{split}-{shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp_path, compression="zstd")
    size_mb = tmp_path.stat().st_size / (1024 * 1024)
    print(f"  Wrote {filename} ({len(rows)} rows, {size_mb:.1f}MB)")

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(tmp_path),
        path_in_repo=f"features/{filename}",
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"Add features shard {filename}",
    )
    print(f"  Uploaded {filename}")
    tmp_path.unlink()  # free disk immediately


# ---------------------------------------------------------------------------
# Streaming processors
# ---------------------------------------------------------------------------

def process_egmd_streaming(
    repo: str,
    token: str,
    max_shard_bytes: int,
) -> int:
    """Stream E-GMD Parquet rows, compute features, write sharded Parquet.

    Returns total number of feature rows written.
    """
    from datasets import Audio, load_dataset

    total = 0
    for split in ["train", "validation", "test"]:
        print(f"\n=== {split} ===")
        try:
            ds = load_dataset(repo, split=split, token=token, streaming=True)
            # Disable automatic audio decoding — we decode with soundfile ourselves
            ds = ds.cast_column("audio", Audio(decode=False))
        except Exception as e:
            print(f"  Split '{split}' not found, skipping: {e}")
            continue

        rows: list[dict] = []
        current_bytes = 0
        shard_idx = 0
        split_rows = 0

        for i, item in enumerate(ds):
            try:
                # Decode audio from bytes
                audio_bytes = item["audio"]["bytes"]
                audio = load_audio_from_bytes(audio_bytes)

                mel = compute_mel(audio)
                n_frames = mel.shape[1]

                # Parse MIDI from bytes
                midi_bytes = item["midi"]
                if isinstance(midi_bytes, dict):
                    midi_bytes = midi_bytes["bytes"]
                events = parse_midi_from_bytes(midi_bytes)

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
                    "split": split,
                    "style": item.get("style", "") or "",
                    "bpm": int(item.get("bpm", 0) or 0),
                    "drummer": item.get("drummer", "") or "",
                    "session": item.get("session", "") or "",
                    "beat_type": item.get("beat_type", "") or "",
                    "time_signature": item.get("time_signature", "") or "",
                    "kit_name": item.get("kit_name", "") or "",
                    "source_audio": item.get("filename", "") or "",
                }

                row_bytes = (
                    len(row["mel_spectrogram"])
                    + len(row["onset_targets"])
                    + len(row["velocity_targets"])
                )
                rows.append(row)
                current_bytes += row_bytes
                split_rows += 1

                # Flush shard when big enough
                if current_bytes >= max_shard_bytes:
                    _write_and_upload_shard(rows, split, shard_idx, repo, token)
                    shard_idx += 1
                    rows = []
                    current_bytes = 0

            except Exception as e:
                print(f"  SKIP row {i}: {e}", file=sys.stderr)

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1} rows ({split_rows} features so far)")

        # Flush remaining rows
        if rows:
            _write_and_upload_shard(rows, split, shard_idx, repo, token)
            shard_idx += 1

        print(f"  {split}: {split_rows} features in {shard_idx} shards")
        total += split_rows

    return total


def process_star_streaming(
    repo: str,
    token: str,
    max_shard_bytes: int,
) -> int:
    """Download STAR files one at a time, compute features, write Parquet.

    Returns total number of feature rows written.
    """
    api = HfApi(token=token)
    total = 0

    for split_name in ["training", "validation", "test"]:
        split_key = "train" if split_name == "training" else split_name
        print(f"\n=== {split_key} ===")

        # List all files under this split
        try:
            all_files = list(api.list_repo_tree(
                repo, repo_type="dataset",
                path_in_repo=f"data/{split_name}",
                recursive=True,
            ))
        except Exception as e:
            print(f"  Split '{split_name}' not found: {e}")
            continue

        # Find annotation .txt files and build pairs
        ann_files = [
            f.path for f in all_files
            if hasattr(f, "path") and f.path.endswith(".txt")
            and "/annotation/" in f.path
        ]

        if not ann_files:
            print(f"  No annotation files found for {split_name}")
            continue

        print(f"  Found {len(ann_files)} annotation files")

        # Build a set of available audio files for fast lookup
        audio_files_set = {
            f.path for f in all_files
            if hasattr(f, "path")
            and "/audio/mix/" in f.path
            and (f.path.endswith(".flac") or f.path.endswith(".wav")
                 or f.path.endswith(".mp3"))
        }

        rows: list[dict] = []
        current_bytes = 0
        shard_idx = 0
        split_rows = 0

        for i, ann_path in enumerate(sorted(ann_files)):
            # Derive expected audio path from annotation path
            # e.g., data/training/source/annotation/track.txt
            #     -> data/training/source/audio/mix/track.flac
            stem = Path(ann_path).stem
            ann_dir = str(Path(ann_path).parent)
            # Go from .../annotation/ to .../audio/mix/
            base_dir = ann_dir.rsplit("/annotation", 1)[0]
            audio_path = None
            for ext in [".flac", ".wav", ".mp3"]:
                candidate = f"{base_dir}/audio/mix/{stem}{ext}"
                if candidate in audio_files_set:
                    audio_path = candidate
                    break

            if audio_path is None:
                print(f"  SKIP {ann_path}: no matching audio file",
                      file=sys.stderr)
                continue

            try:
                # Download both files
                local_ann = hf_hub_download(
                    repo_id=repo, repo_type="dataset",
                    filename=ann_path, token=token,
                )
                local_audio = hf_hub_download(
                    repo_id=repo, repo_type="dataset",
                    filename=audio_path, token=token,
                )

                # Process
                audio = load_audio_mono(local_audio)
                mel = compute_mel(audio)
                n_frames = mel.shape[1]
                events = parse_star_annotation(local_ann)
                onset, vel = compute_onset_targets(events, n_frames)

                # Extract source directory name
                # e.g., data/training/MDB_Drums/annotation/...
                parts = ann_path.split("/")
                source = parts[2] if len(parts) > 2 else ""

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
                    "split": split_key,
                    "source": source,
                    "track_id": stem,
                    "source_audio": Path(audio_path).name,
                }

                row_bytes = (
                    len(row["mel_spectrogram"])
                    + len(row["onset_targets"])
                    + len(row["velocity_targets"])
                )
                rows.append(row)
                current_bytes += row_bytes
                split_rows += 1

                # Flush shard when big enough
                if current_bytes >= max_shard_bytes:
                    _write_and_upload_shard(rows, split_key, shard_idx, repo, token)
                    shard_idx += 1
                    rows = []
                    current_bytes = 0

                # Clean up downloaded files to save disk space
                try:
                    os.remove(local_ann)
                except OSError:
                    pass
                try:
                    os.remove(local_audio)
                except OSError:
                    pass

            except Exception as e:
                print(f"  SKIP {ann_path}: {e}", file=sys.stderr)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(ann_files)} "
                      f"({split_rows} features so far)")

        # Flush remaining rows
        if rows:
            _write_and_upload_shard(rows, split_key, shard_idx, repo, token)
            shard_idx += 1

        print(f"  {split_key}: {split_rows} features in {shard_idx} shards")
        total += split_rows

    return total


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

    max_shard_bytes = args.max_shard_mb * 1024 * 1024

    print(f"Dataset:  {args.dataset}")
    print(f"Repo:     {args.repo}")
    print(f"Shard MB: {args.max_shard_mb}")
    print(f"Params:   sr={SAMPLE_RATE}, hop={HOP_LENGTH}, n_mels={N_MELS}, "
          f"fps={FPS}, classes={NUM_CLASSES}")
    print()

    t0 = time.monotonic()

    if args.dataset == "egmd":
        total = process_egmd_streaming(
            args.repo, token, max_shard_bytes,
        )
    elif args.dataset == "star":
        total = process_star_streaming(
            args.repo, token, max_shard_bytes,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    elapsed = time.monotonic() - t0
    print(f"\nFeature computation done: {total} rows ({elapsed:.1f}s)")

    if total == 0:
        print("No features computed. Check dataset contents.", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll done. {total} feature rows uploaded to "
          f"{args.repo} (features/ config).")


if __name__ == "__main__":
    main()
