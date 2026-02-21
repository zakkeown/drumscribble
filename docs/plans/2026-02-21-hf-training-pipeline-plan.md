# HF Training Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run DrumscribbleCNN training on HF Jobs using pre-computed features with Trackio experiment tracking and validation F1 every 10 epochs.

**Architecture:** Two-phase pipeline. Phase 1: HF Job computes mel spectrograms + onset/velocity targets from raw datasets, uploads as Parquet "features" config. Phase 2: Training job loads pre-computed features via HF `datasets` library, trains with Trackio logging, evaluates val F1 every 10 epochs, uploads checkpoints to `schismaudio/drumscribble-checkpoints`.

**Tech Stack:** PyTorch, HF `datasets`, HF `huggingface_hub`, Trackio, HF Jobs (PEP 723 UV scripts), `pyarrow`

---

### Task 1: FeaturesDataset — Failing Tests

**Files:**
- Create: `tests/test_features.py`

Write tests for a `FeaturesDataset` class that loads pre-computed features from a dict-like source (simulating HF datasets rows), random-crops to a chunk length, and returns `(mel, onset_target, vel_target)` tensors matching the existing dataset interface.

**Step 1: Write the failing tests**

```python
"""Tests for FeaturesDataset (pre-computed features from HF datasets)."""

import numpy as np
import pytest
import torch

from drumscribble.config import NUM_CLASSES, N_MELS
from drumscribble.data.features import FeaturesDataset


def _make_feature_row(n_frames: int = 1000) -> dict:
    """Create a fake feature row matching the Parquet schema."""
    mel = np.random.randn(N_MELS, n_frames).astype(np.float32)
    onset = np.random.rand(NUM_CLASSES, n_frames).astype(np.float32)
    vel = np.random.rand(NUM_CLASSES, n_frames).astype(np.float32)
    return {
        "mel_spectrogram": mel.tobytes(),
        "onset_targets": onset.tobytes(),
        "velocity_targets": vel.tobytes(),
        "n_frames": n_frames,
        "n_mels": N_MELS,
        "n_classes": NUM_CLASSES,
        "duration": n_frames / 62.5,
        "split": "train",
    }


class TestFeaturesDataset:
    def test_getitem_returns_correct_shapes(self):
        rows = [_make_feature_row(1000) for _ in range(5)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        mel, onset, vel = ds[0]

        assert mel.shape == (1, N_MELS, 625)
        assert onset.shape == (NUM_CLASSES, 625)
        assert vel.shape == (NUM_CLASSES, 625)

    def test_getitem_returns_tensors(self):
        rows = [_make_feature_row(1000)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        mel, onset, vel = ds[0]

        assert isinstance(mel, torch.Tensor)
        assert isinstance(onset, torch.Tensor)
        assert isinstance(vel, torch.Tensor)
        assert mel.dtype == torch.float32

    def test_length_counts_chunks(self):
        # 1000 frames / 625 chunk = 1 full chunk per row (non-overlapping)
        rows = [_make_feature_row(1000) for _ in range(3)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        assert len(ds) == 3  # 1 chunk per row

    def test_multiple_chunks_per_row(self):
        # 2000 frames / 625 = 3 full chunks
        rows = [_make_feature_row(2000)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        assert len(ds) == 3

    def test_skips_short_rows(self):
        rows = [
            _make_feature_row(100),   # too short for 625 chunk
            _make_feature_row(1000),  # 1 chunk
        ]
        ds = FeaturesDataset(rows, chunk_frames=625)
        assert len(ds) == 1

    def test_random_crop_varies(self):
        rows = [_make_feature_row(2000)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        mel1, _, _ = ds[0]
        mel2, _, _ = ds[0]
        # Random crop means different calls may give different data
        # (not guaranteed, but with 2000 frames and 625 crop, very likely)
        # Just verify shapes are correct
        assert mel1.shape == mel2.shape == (1, N_MELS, 625)

    def test_works_with_dataloader(self):
        rows = [_make_feature_row(1000) for _ in range(4)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        batch = next(iter(loader))
        mel, onset, vel = batch
        assert mel.shape == (2, 1, N_MELS, 625)
        assert onset.shape == (2, NUM_CLASSES, 625)
        assert vel.shape == (2, NUM_CLASSES, 625)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'drumscribble.data.features'`

**Step 3: Commit**

```bash
git add tests/test_features.py
git commit -m "test: add failing tests for FeaturesDataset"
```

---

### Task 2: FeaturesDataset — Implementation

**Files:**
- Create: `src/drumscribble/data/features.py`

Implement `FeaturesDataset` to pass all tests from Task 1.

**Step 1: Write the implementation**

```python
"""Dataset that loads pre-computed features from HF datasets Parquet rows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from drumscribble.config import N_MELS, NUM_CLASSES


class FeaturesDataset(Dataset):
    """Load pre-computed mel spectrograms and onset/velocity targets.

    Each row contains full-length features (not chunked). This dataset
    builds a chunk index for non-overlapping windows and applies a random
    offset within each window on __getitem__.

    Args:
        rows: List of dicts with keys: mel_spectrogram (bytes),
              onset_targets (bytes), velocity_targets (bytes), n_frames (int).
        chunk_frames: Number of frames per training chunk (default 625 = 10s at 62.5fps).
    """

    def __init__(self, rows: list[dict], chunk_frames: int = 625) -> None:
        self.chunk_frames = chunk_frames
        self.rows = []
        self.chunks: list[tuple[int, int]] = []  # (row_idx, start_frame)

        for i, row in enumerate(rows):
            n_frames = row["n_frames"]
            if n_frames < chunk_frames:
                continue
            self.rows.append(row)
            row_idx = len(self.rows) - 1
            n_chunks = n_frames // chunk_frames
            for c in range(n_chunks):
                self.chunks.append((row_idx, c * chunk_frames))

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row_idx, base_start = self.chunks[idx]
        row = self.rows[row_idx]
        n_frames = row["n_frames"]

        # Random offset within this chunk's window, clamped to valid range
        max_start = min(base_start + self.chunk_frames, n_frames - self.chunk_frames)
        start = torch.randint(base_start, max_start + 1, (1,)).item() if max_start > base_start else base_start

        mel = np.frombuffer(row["mel_spectrogram"], dtype=np.float32).reshape(N_MELS, n_frames)
        onset = np.frombuffer(row["onset_targets"], dtype=np.float32).reshape(NUM_CLASSES, n_frames)
        vel = np.frombuffer(row["velocity_targets"], dtype=np.float32).reshape(NUM_CLASSES, n_frames)

        mel_chunk = torch.from_numpy(mel[:, start:start + self.chunk_frames].copy()).unsqueeze(0)
        onset_chunk = torch.from_numpy(onset[:, start:start + self.chunk_frames].copy())
        vel_chunk = torch.from_numpy(vel[:, start:start + self.chunk_frames].copy())

        return mel_chunk, onset_chunk, vel_chunk
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/test_features.py -v`
Expected: All 7 tests PASS

**Step 3: Commit**

```bash
git add src/drumscribble/data/features.py
git commit -m "feat: add FeaturesDataset for pre-computed HF features"
```

---

### Task 3: Compute Features HF Job Script

**Files:**
- Create: `scripts/compute_features_job.py`

Adapt existing `scripts/hf_upload/compute_features.py` into a PEP 723 HF Jobs script that downloads a dataset from HF Hub, computes features, and uploads them back as a `features/` config.

**Step 1: Write the script**

```python
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
"""HF Jobs script: compute mel + onset features and upload as dataset config.

Downloads raw audio dataset from HF Hub, computes mel spectrograms and
onset/velocity targets, writes sharded Parquet, uploads as "features" config.

Usage (via hf jobs):
    # E-GMD
    hf jobs uv run scripts/compute_features_job.py \
        --flavor cpu-upgrade --timeout 6h \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --dataset egmd --repo schismaudio/e-gmd

    # STAR (private)
    hf jobs uv run scripts/compute_features_job.py \
        --flavor cpu-upgrade --timeout 6h \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --dataset star --repo zkeown/star-drums
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pretty_midi
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download

# ---------------------------------------------------------------------------
# Audio & target parameters (match drumscribble.config)
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
HOP_LENGTH = 256
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

EGMD_NOTE_REMAP = {
    22: 42, 26: 46, 58: 43,
}

STAR_ABBREV_TO_GM = {
    "BD": 36, "SD": 38, "CHH": 42, "PHH": 44, "OHH": 46,
    "HT": 48, "MT": 45, "LT": 43, "CRC": 49, "SPC": 55,
    "CHC": 52, "RD": 51, "RB": 53, "CB": 56, "CL": 75,
    "CLP": 39, "SS": 37, "TB": 54,
}


# ---------------------------------------------------------------------------
# Feature computation (same logic as compute_features.py)
# ---------------------------------------------------------------------------

def load_audio_mono(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def compute_mel(audio: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX, power=2.0,
    )
    return np.log(np.maximum(mel, 1e-7)).astype(np.float32)


def compute_onset_targets(events, n_frames):
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


def parse_midi_events(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    events = []
    for inst in pm.instruments:
        if inst.is_drum:
            for note in inst.notes:
                events.append((note.start, note.pitch, note.velocity))
    events.sort(key=lambda x: x[0])
    return events


def parse_star_annotation(ann_path):
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
# Dataset entry discovery
# ---------------------------------------------------------------------------

def discover_egmd(root):
    csv_path = None
    for name in ["e-gmd-v1.0.0.csv", "info.csv"]:
        candidate = root / name
        if candidate.exists():
            csv_path = candidate
            break
    if csv_path is None:
        raise FileNotFoundError(f"No CSV manifest in {root}")
    entries = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            audio = root / row["audio_filename"]
            midi = root / row["midi_filename"]
            if audio.exists() and midi.exists():
                entries.append({
                    "audio_path": str(audio), "midi_path": str(midi),
                    "split": row.get("split", "train"),
                    "duration": float(row.get("duration", 0)),
                    "style": row.get("style", ""),
                    "bpm": int(row["bpm"]) if row.get("bpm") else 0,
                    "drummer": row.get("drummer", ""),
                    "session": row.get("session", ""),
                })
    return entries


def discover_star(root):
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
                for ext in [".flac", ".wav", ".mp3"]:
                    candidate = audio_dir / (ann_file.stem + ext)
                    if candidate.exists():
                        split = "train" if split_name == "training" else split_name
                        entries.append({
                            "audio_path": str(candidate),
                            "annotation_path": str(ann_file),
                            "split": split,
                            "source": source_dir.name,
                            "track_id": ann_file.stem,
                        })
                        break
    return entries


# ---------------------------------------------------------------------------
# Process entries -> feature rows
# ---------------------------------------------------------------------------

def process_egmd_entry(entry):
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
            "n_frames": n_frames, "n_mels": N_MELS, "n_classes": NUM_CLASSES,
            "sample_rate": SAMPLE_RATE, "hop_length": HOP_LENGTH, "fps": FPS,
            "duration": len(audio) / SAMPLE_RATE,
            "split": entry["split"],
            "style": entry.get("style", ""),
            "bpm": entry.get("bpm", 0),
            "drummer": entry.get("drummer", ""),
            "session": entry.get("session", ""),
            "source_audio": Path(entry["audio_path"]).name,
        }
    except Exception as e:
        print(f"  SKIP {entry['audio_path']}: {e}", file=sys.stderr)
        return None


def process_star_entry(entry):
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
            "n_frames": n_frames, "n_mels": N_MELS, "n_classes": NUM_CLASSES,
            "sample_rate": SAMPLE_RATE, "hop_length": HOP_LENGTH, "fps": FPS,
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

def write_parquet_shards(rows, output_dir, split, max_shard_bytes=500 * 1024 * 1024):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  No rows for split '{split}', skipping.")
        return

    shards = []
    current_shard, current_bytes = [], 0
    for row in rows:
        row_bytes = len(row["mel_spectrogram"]) + len(row["onset_targets"]) + len(row["velocity_targets"])
        if current_bytes + row_bytes > max_shard_bytes and current_shard:
            shards.append(current_shard)
            current_shard, current_bytes = [], 0
        current_shard.append(row)
        current_bytes += row_bytes
    if current_shard:
        shards.append(current_shard)

    for shard_idx, shard_rows in enumerate(shards):
        filename = f"{split}-{shard_idx:05d}-of-{len(shards):05d}.parquet"
        table = pa.Table.from_pylist(shard_rows)
        pq.write_table(table, output_dir / filename, compression="zstd")
        print(f"  Wrote {filename} ({len(shard_rows)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute features and upload to HF")
    parser.add_argument("--dataset", required=True, choices=["egmd", "star"])
    parser.add_argument("--repo", required=True, help="HF dataset repo (e.g. schismaudio/e-gmd)")
    parser.add_argument("--max-shard-mb", type=int, default=500)
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    api = HfApi(token=token)

    # Download raw dataset
    print(f"Downloading {args.repo}...")
    data_dir = snapshot_download(
        repo_id=args.repo, repo_type="dataset", token=token,
        local_dir=f"/tmp/{args.dataset}",
    )
    data_root = Path(data_dir)
    print(f"Downloaded to {data_root}")

    # Discover entries
    if args.dataset == "egmd":
        entries = discover_egmd(data_root)
        process_fn = process_egmd_entry
    else:
        entries = discover_star(data_root)
        process_fn = process_star_entry

    print(f"Found {len(entries)} entries")

    # Group by split
    splits = {}
    for entry in entries:
        splits.setdefault(entry["split"], []).append(entry)
    for s, e in sorted(splits.items()):
        print(f"  {s}: {len(e)} entries")

    # Process and write
    output_dir = Path(f"/tmp/{args.dataset}_features")
    max_shard_bytes = args.max_shard_mb * 1024 * 1024
    total = 0

    for split_name, split_entries in sorted(splits.items()):
        print(f"\n=== {split_name} ({len(split_entries)} entries) ===")
        rows = []
        for i, entry in enumerate(split_entries):
            row = process_fn(entry)
            if row is not None:
                rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(split_entries)} processed ({len(rows)} ok)")
        print(f"  {len(rows)} features computed")
        write_parquet_shards(rows, output_dir, split_name, max_shard_bytes)
        total += len(rows)

    print(f"\nTotal: {total} features")

    # Upload as "features/" config
    print(f"\nUploading features to {args.repo} (features/ config)...")
    api.upload_folder(
        repo_id=args.repo, repo_type="dataset", token=token,
        folder_path=str(output_dir),
        path_in_repo="features",
    )
    print(f"Upload complete: {args.repo}/features")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script syntax**

Run: `python -c "import ast; ast.parse(open('scripts/compute_features_job.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/compute_features_job.py
git commit -m "feat: add HF Jobs script for computing dataset features"
```

---

### Task 4: Update train_hf_job.py — Add Trackio + FeaturesDataset + Val F1

**Files:**
- Modify: `scripts/train_hf_job.py`

This is the largest task. Update the training script to:
1. Use `FeaturesDataset` with HF `datasets` library instead of raw audio
2. Point to new repos (`schismaudio/e-gmd`, `zkeown/star-drums`)
3. Output to `schismaudio/drumscribble-checkpoints`
4. Add Trackio experiment tracking
5. Add validation F1 evaluation every 10 epochs

**Step 1: Rewrite the training script**

Replace the full contents of `scripts/train_hf_job.py` with:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "drumscribble @ git+https://github.com/zakkeown/drumscribble.git",
#     "huggingface_hub[hf_xet]",
#     "datasets",
#     "trackio",
#     "pyyaml",
# ]
# ///
"""HF Jobs training script for DrumscribbleCNN.

Loads pre-computed features from HF datasets, trains DrumscribbleCNN,
logs metrics via Trackio, evaluates val F1 every 10 epochs, and uploads
checkpoints to HF Hub.

Usage (via hf jobs):
    # E-GMD only
    hf jobs uv run scripts/train_hf_job.py \
        --flavor a10g-large --timeout 48h \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --dataset egmd --epochs 100

    # Multi-dataset (E-GMD + STAR)
    hf jobs uv run scripts/train_hf_job.py \
        --flavor a10g-large --timeout 48h \
        --secret HF_TOKEN=$HF_TOKEN \
        -- --dataset multi --epochs 100 --run-name baseline-v1
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import trackio
from datasets import load_dataset
from huggingface_hub import HfApi


def load_features(repo_id: str, split: str, token: str) -> list[dict]:
    """Load pre-computed features from HF dataset repo."""
    print(f"Loading features from {repo_id} (split={split})...")
    ds = load_dataset(repo_id, "features", split=split, token=token)
    rows = list(ds)
    print(f"  Loaded {len(rows)} rows")
    return rows


@torch.no_grad()
def validate(model, val_rows, device, chunk_frames=625):
    """Run validation: compute onset F1 on validation set."""
    from drumscribble.config import FPS, GM_CLASSES
    from drumscribble.evaluate import evaluate_events
    from drumscribble.inference import detections_to_events
    from drumscribble.data.features import FeaturesDataset

    val_ds = FeaturesDataset(val_rows, chunk_frames=chunk_frames)
    loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    all_ref = []
    all_est = []
    model.eval()

    for mel, onset_target, vel_target in loader:
        mel = mel.to(device)
        with torch.amp.autocast("cuda", enabled=device == "cuda"):
            onset_pred, vel_pred, _ = model(mel)

        # Process each item in batch
        for b in range(mel.shape[0]):
            # Reference events from targets
            ref_events = []
            ot = onset_target[b]  # (26, T)
            for cls_idx in range(ot.shape[0]):
                peaks = (ot[cls_idx] >= 1.0).nonzero(as_tuple=True)[0]
                for p in peaks:
                    ref_events.append({
                        "time": p.item() / FPS,
                        "note": GM_CLASSES[cls_idx],
                    })

            # Estimated events from predictions
            est_events = detections_to_events(
                onset_pred[b].cpu(), vel_pred[b].cpu(),
                threshold=0.5, fps=FPS,
            )

            all_ref.append(ref_events)
            all_est.append(est_events)

    model.train()

    # Micro-average across all examples
    total_tp, total_fp, total_fn = 0, 0, 0
    for ref, est in zip(all_ref, all_est):
        metrics = evaluate_events(ref, est)
        # Approximate TP/FP/FN from precision/recall
        n_est = len(est) or 1
        n_ref = len(ref) or 1
        tp = metrics["precision"] * n_est
        total_tp += tp
        total_fp += n_est - tp
        total_fn += n_ref - metrics["recall"] * n_ref

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"val_f1": f1, "val_precision": precision, "val_recall": recall}


def main():
    parser = argparse.ArgumentParser(description="Train DrumscribbleCNN on HF Jobs")
    parser.add_argument("--dataset", type=str, default="egmd",
                        choices=["egmd", "star", "multi"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--egmd-repo", type=str, default="schismaudio/e-gmd")
    parser.add_argument("--star-repo", type=str, default="zkeown/star-drums")
    parser.add_argument("--output-repo", type=str, default="schismaudio/drumscribble-checkpoints")
    parser.add_argument("--dataset-weights", type=float, nargs=2, default=[0.5, 0.5])
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None,
                        help="Trackio run name (e.g. baseline-v1)")
    parser.add_argument("--trackio-space", type=str, default="schismaudio/trackio",
                        help="HF Space for Trackio dashboard")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        sys.exit(1)

    api = HfApi(token=token)

    # --- Device ---
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("WARNING: No GPU detected, training on CPU")

    # --- Trackio ---
    from drumscribble.config import FPS
    chunk_frames = int(args.chunk_seconds * FPS)

    trackio.init(
        project="drumscribble",
        name=args.run_name,
        space_id=args.trackio_space,
        config={
            "dataset": args.dataset,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "chunk_seconds": args.chunk_seconds,
            "dataset_weights": args.dataset_weights,
        },
    )

    # --- Load features ---
    from drumscribble.data.features import FeaturesDataset

    train_rows = []
    val_rows = []

    if args.dataset in ("egmd", "multi"):
        train_rows += load_features(args.egmd_repo, "train", token)
        val_rows += load_features(args.egmd_repo, "validation", token)

    if args.dataset in ("star", "multi"):
        train_rows += load_features(args.star_repo, "train", token)
        val_rows += load_features(args.star_repo, "validation", token)

    train_ds = FeaturesDataset(train_rows, chunk_frames=chunk_frames)
    print(f"Training chunks: {len(train_ds):,}")
    print(f"Validation rows: {len(val_rows):,}")

    # --- Create output repo if needed ---
    try:
        api.repo_info(repo_id=args.output_repo, repo_type="model")
    except Exception:
        print(f"Creating output repo {args.output_repo}...")
        api.create_repo(repo_id=args.output_repo, repo_type="model", private=True)

    # --- Import training components ---
    from torch.utils.data import DataLoader

    from drumscribble.data.augment import SpecAugment
    from drumscribble.loss import DrumscribbleLoss
    from drumscribble.model.drumscribble import DrumscribbleCNN
    from drumscribble.train import (
        EMAModel,
        create_optimizer,
        create_scheduler,
        train_one_epoch,
    )

    # --- Picklable collate ---
    class AugmentCollate:
        def __init__(self, augment):
            self.augment = augment

        def __call__(self, batch):
            mels, onsets, vels = zip(*batch)
            mel_batch = torch.stack(mels)
            onset_batch = torch.stack(onsets)
            vel_batch = torch.stack(vels)
            mel_batch = self.augment(mel_batch)
            return mel_batch, onset_batch, vel_batch

    # --- Model ---
    model = DrumscribbleCNN(
        backbone_dims=(64, 128, 256, 384),
        backbone_depths=(5, 5, 5, 5),
        num_attn_layers=3,
        num_attn_heads=4,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # --- DataLoader ---
    augment = SpecAugment()
    collate_fn = AugmentCollate(augment)

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn,
    )

    # --- Optimizer ---
    optimizer = create_optimizer(model, lr=args.lr, weight_decay=0.05)
    loss_fn = DrumscribbleLoss()

    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(loader)
    total_steps = args.epochs * len(loader)
    scheduler = create_scheduler(optimizer, warmup_steps, total_steps)

    ema = EMAModel(model, decay=0.999)

    # --- AMP ---
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    if scaler:
        print("Using CUDA AMP")

    # --- Resume ---
    start_epoch = 0
    if args.resume_from:
        print(f"Downloading checkpoint {args.resume_from}...")
        ckpt_path = api.hf_hub_download(
            repo_id=args.output_repo, filename=args.resume_from, token=token,
        )
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch}")

    # --- Training loop ---
    output_dir = Path("/tmp/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining: {args.dataset} | epochs {start_epoch+1}-{args.epochs} | "
          f"batch_size={args.batch_size} | lr={args.lr}")
    print(f"Steps/epoch: {len(loader):,} | total steps: {total_steps:,}")
    print("=" * 60)

    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, args.epochs):
        avg_loss = train_one_epoch(
            model, loader, optimizer, loss_fn,
            device=device, scheduler=scheduler, scaler=scaler, ema=ema,
        )
        global_step += len(loader)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

        trackio.log({"loss": avg_loss, "lr": lr_now, "epoch": epoch + 1})

        # Checkpoint + validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Validation
            ema.apply(model)
            val_metrics = validate(model, val_rows, device, chunk_frames)
            ema.restore(model)

            print(f"  Val F1={val_metrics['val_f1']:.4f} "
                  f"P={val_metrics['val_precision']:.4f} "
                  f"R={val_metrics['val_recall']:.4f}")
            trackio.log(val_metrics)

            # Save checkpoint
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            ckpt_data = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "val_f1": val_metrics["val_f1"],
            }
            if scaler is not None:
                ckpt_data["scaler"] = scaler.state_dict()
            torch.save(ckpt_data, ckpt_path)

            api.upload_file(
                path_or_fileobj=str(ckpt_path),
                path_in_repo=f"checkpoint_epoch{epoch+1}.pt",
                repo_id=args.output_repo, repo_type="model", token=token,
            )
            print(f"  Uploaded checkpoint to {args.output_repo}")

    # --- Final with EMA ---
    ema.apply(model)
    final_path = output_dir / "final.pt"
    torch.save({"model": model.state_dict(), "epoch": args.epochs}, final_path)
    ema.restore(model)

    api.upload_file(
        path_or_fileobj=str(final_path), path_in_repo="final.pt",
        repo_id=args.output_repo, repo_type="model", token=token,
    )

    trackio.finish()
    print(f"\nTraining complete! Final checkpoint (EMA) uploaded to {args.output_repo}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script syntax**

Run: `python -c "import ast; ast.parse(open('scripts/train_hf_job.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add scripts/train_hf_job.py
git commit -m "feat: update training script for pre-computed features, Trackio, val F1"
```

---

### Task 5: Add datasets + trackio to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

Add `datasets` and `trackio` as optional dependencies under a new `[project.optional-dependencies.hf]` group.

**Step 1: Add the dependency group**

Add to `pyproject.toml` optional-dependencies:

```toml
hf = [
    "datasets>=3.0",
    "trackio>=0.1",
    "huggingface_hub[hf_xet]>=0.28",
]
```

**Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add HF training optional dependencies"
```

---

### Task 6: Run Existing Tests — Verify No Regressions

**Files:** None (verification only)

**Step 1: Run the full test suite**

Run: `pytest tests/ -v --ignore=tests/test_export.py --ignore=tests/test_mert.py`

(Ignore export and mert tests — they require coremltools and transformers which may not be installed.)

Expected: All existing tests PASS. New `test_features.py` tests PASS.

**Step 2: Run features tests specifically**

Run: `pytest tests/test_features.py -v`
Expected: All 7 tests PASS

---

### Task 7: Commit All and Verify

**Step 1: Check git status**

Run: `git status`
Expected: Clean working tree (all changes committed in prior tasks)

**Step 2: Run full test suite one more time**

Run: `pytest tests/ -v --ignore=tests/test_export.py --ignore=tests/test_mert.py`
Expected: All PASS

---

## Execution Commands Reference

After implementation, run the actual HF Jobs:

```bash
# Step 1: Compute E-GMD features
hf jobs uv run scripts/compute_features_job.py \
    --flavor cpu-upgrade --timeout 6h \
    --secret HF_TOKEN=$HF_TOKEN \
    -- --dataset egmd --repo schismaudio/e-gmd

# Step 2: Compute STAR features
hf jobs uv run scripts/compute_features_job.py \
    --flavor cpu-upgrade --timeout 6h \
    --secret HF_TOKEN=$HF_TOKEN \
    -- --dataset star --repo zkeown/star-drums

# Step 3: Baseline training
hf jobs uv run scripts/train_hf_job.py \
    --flavor a10g-large --timeout 48h \
    --secret HF_TOKEN=$HF_TOKEN \
    -- --dataset multi --epochs 100 --run-name baseline-v1
```
