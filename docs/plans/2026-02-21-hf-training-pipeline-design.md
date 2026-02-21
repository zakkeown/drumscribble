# HF Training Pipeline Design

**Date:** 2026-02-21
**Status:** Approved

## Goal

Run DrumscribbleCNN training on HF Jobs infrastructure using pre-computed features for fast iteration. Baseline first, then experiment.

## Architecture

Two-phase system:

**Phase 1 — Feature Computation (one-time per dataset)**
- HF Job downloads raw dataset from HF Hub
- Computes mel spectrograms (128 bins, 16kHz, 256 hop) + onset/velocity targets
- Writes sharded Parquet files (500MB, zstd) as a `features/` config in the dataset repo
- Runs on CPU instance (~2-3h per dataset)

**Phase 2 — Training (repeated per experiment)**
- HF Job downloads pre-computed features (~5-10GB vs ~90GB raw)
- Loads via `datasets.load_dataset()`, random crops to chunk length, applies SpecAugment
- Trains DrumscribbleCNN on A10G GPU
- Uploads checkpoints every 10 epochs to `schismaudio/drumscribble-checkpoints`
- Logs metrics via Trackio to HF Space dashboard
- Runs validation F1 every 10 epochs (aligned with checkpoint saves)

## Dataset Sources

| Dataset | Repo | Visibility | Raw Size |
|---------|------|-----------|----------|
| E-GMD | `schismaudio/e-gmd` | Public | ~90GB |
| STAR Drums | `zkeown/star-drums` | Private | TBD |

Training uses multi-dataset mode: `ConcatDataset` + `WeightedRandomSampler` (50/50 weight).

## Feature Schema

Full-length recordings stored per row (not chunked). Training dataloader handles random cropping.

| Column | Type | Description |
|--------|------|-------------|
| `mel_spectrogram` | bytes | `(128, T)` float32 log-mel, raw numpy bytes |
| `onset_targets` | bytes | `(26, T)` float32 onset targets with widening |
| `velocity_targets` | bytes | `(26, T)` float32 velocity at peak frames |
| `n_frames` | int | Number of spectrogram frames |
| `duration` | float | Audio duration in seconds |
| `split` | string | train/validation/test |
| + metadata | various | style, bpm, drummer, source, etc. |

## Data Pipeline (Training)

```
load_dataset(repo, "features", split="train")
  → Parquet download
  → Deserialize bytes → numpy
  → Random crop to chunk_frames (625 for 10s)
  → Convert to torch tensors
  → SpecAugment in collate_fn
  → Model forward/backward
```

## Trackio Integration

```python
trackio.init(project="drumscribble", run="baseline-v1")
trackio.log(step=global_step, loss=loss, lr=lr)
trackio.log(step=epoch, val_f1=f1, val_precision=prec, val_recall=rec)
```

## Validation

Every 10 epochs (aligned with checkpoint saves):
1. Run inference on validation split (full-length, overlapping windows)
2. Compute onset F1 via mir_eval
3. Log val_f1, val_precision, val_recall to Trackio

## Hardware

- Feature computation: `cpu-basic` flavor, 6h timeout
- Training: `a10g-large` (24GB VRAM), 48h timeout
- Batch size 32, AdamW, cosine LR with 5-epoch warmup, EMA decay=0.999

## Deliverables

| # | File | Description |
|---|------|-------------|
| 1 | `scripts/compute_features_job.py` | PEP 723 HF Jobs script — compute features, upload as config |
| 2 | `src/drumscribble/data/features.py` | `FeaturesDataset` — loads features config, random crop, tensors |
| 3 | `scripts/train_hf_job.py` (updated) | Use FeaturesDataset, Trackio, val F1, schismaudio repos |
| 4 | Tests for `FeaturesDataset` | Deserialization, cropping, tensor shapes |

## What Stays the Same

Model architecture, loss function, optimizer/scheduler, EMA, AMP, SpecAugment, checkpoint format. No changes to model/, loss.py, train.py, evaluate.py, inference.py.

## Execution Order

1. Compute E-GMD features (HF Job, CPU, ~2-3h)
2. Compute STAR features (HF Job, CPU, ~1-2h)
3. Run baseline training (HF Job, A10G, ~24-48h)
4. Iterate experiments
