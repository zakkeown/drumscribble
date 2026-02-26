# Parquet Mel Spectrogram Dataset — Design Document

**Date**: 2026-02-25
**Status**: Approved

---

## Problem

The current training pipeline computes mel spectrograms and target frames on every `__getitem__` call — loading raw audio, resampling, running MelSpectrogram, parsing MIDI/TSV annotations, and generating widened targets. This dominates training time, especially on MPS where `num_workers=0`.

Pre-computing everything into a parquet dataset on HF Hub eliminates all I/O and compute overhead at training time. The training loop just reads tensors.

---

## Design

### 1. HF Dataset Repo

**Repo:** `zkeown/drumscribble-mel-specs`

**Schema per row:**

| Column | Type | Flat Length | Reshaped |
|--------|------|-------------|----------|
| `mel` | list\<float32\> | 80,000 | (1, 128, 625) |
| `onset_target` | list\<float32\> | 16,250 | (26, 625) |
| `vel_target` | list\<float32\> | 16,250 | (26, 625) |
| `source` | string | — | `"egmd"` or `"star"` |
| `split` | string | — | `"train"` or `"validation"` |

Stored as HF datasets with native parquet sharding. The `split` column maps to HF dataset splits so `load_dataset(..., split="train")` works directly.

### 2. Preprocessing Script

**`scripts/build_parquet_dataset.py`**

1. Downloads raw E-GMD from Magenta's public URL, extracts
2. Downloads raw STAR from its source, extracts
3. Instantiates existing `EGMDDataset` / `STARDataset` classes
4. Iterates all chunks, flattens mel + targets into rows
5. Builds `datasets.Dataset` with train/validation splits
6. Pushes to `zkeown/drumscribble-mel-specs` via `push_to_hub()`

Reuses existing dataset classes verbatim — no duplicated mel/target logic.

### 3. Training Dataset Wrapper

**`src/drumscribble/data/parquet.py`** — `ParquetDataset`

- `__init__`: calls `datasets.load_dataset(repo, split=split)`, optionally filters by `source`
- `__getitem__`: reshapes flat arrays → tensors, returns `(mel, onset_target, vel_target)`
- Same interface as `EGMDDataset` / `STARDataset`

### 4. CLI Integration

- `--dataset parquet` added to `cli/train.py` choices
- `--hf-dataset` arg (default: `zkeown/drumscribble-mel-specs`)
- `--parquet-source` optional filter (`egmd`, `star`, or both)
- Same options added to `scripts/train_hf_job.py`

### 5. Config

```yaml
data:
  hf_dataset_repo: zkeown/drumscribble-mel-specs
```

---

## What Stays Unchanged

- `train.py` / `train_one_epoch()` — same `(mel, onset, vel)` tuple interface
- `SpecAugment` / `AugmentCollate` — applied via collate_fn as before
- Model, loss, EMA, scheduler, AMP, checkpointing — all untouched
- Existing raw-audio dataset classes — still available for preprocessing or direct use

## New Dependency

`datasets` (HF datasets library) added to `pyproject.toml`.
