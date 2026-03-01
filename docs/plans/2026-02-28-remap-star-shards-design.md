# Design: Remap STAR Feature Shards to 26-Class GM Taxonomy

**Date:** 2026-02-28

## Problem

STAR feature shards store onset/velocity targets with 18 STAR-native classes `(18, T)`. The WebDataset training pipeline expects 26-class GM targets `(26, T)` to match E-GMD. Additionally, STAR onset targets are binary (0/1) while E-GMD uses Gaussian widening `[0.3, 0.6, 1.0, 0.6, 0.3]`.

## Decision

Remap existing feature shards in-place — no audio reprocessing needed since mel spectrograms are class-independent. Only onset/velocity target arrays and params.json change.

## Remapping

STAR's 18 classes map to GM indices via `STAR_ABBREV_TO_GM` + `GM_NOTE_TO_INDEX` from `config.py`:

```
STAR_CLASSES = ["BD", "SD", "CHH", "OHH", "PHH", "HT", "MT", "LT",
                "CRC", "SPC", "CHC", "RD", "RB", "CB", "CL", "CLP", "SS", "TB"]

For each STAR index i:
  gm_note = STAR_ABBREV_TO_GM[STAR_CLASSES[i]]  # e.g. "BD" -> 36
  gm_idx  = GM_NOTE_TO_INDEX[gm_note]            # e.g. 36 -> 1
  new_onset[gm_idx, :] = widen(old_onset[i, :])
  new_vel[gm_idx, :]   = old_vel[i, :]
```

8 GM classes have no STAR equivalent — their rows stay zero.

## Onset Widening

Binary onsets are convolved with `TARGET_WIDENING = [0.3, 0.6, 1.0, 0.6, 0.3]`, clamped to [0, 1]. This matches E-GMD's target format. Velocity targets are not widened — only the center frame carries velocity.

## Data Flow

```
For each split (train, validation, test):
  Read existing feature-shard-*.tar
  For each sample:
    mel_spectrogram.npy (128, T) → copy unchanged
    onset_targets.npy (18, T) → remap to (26, T) + widen
    velocity_targets.npy (18, T) → remap to (26, T)
    params.json → n_classes: 18 → 26
  Write new shards to temp dir
  Swap temp shards over originals
```

## Script

`scripts/remap_star_shards.py` — imports mappings from `drumscribble.config`. Takes `--shard-root` pointing to `~/Documents/Datasets/build/star-drums`.

## Safety

- Writes to a temp directory first, swaps only after all shards in a split are written
- Verifies output shapes before overwriting
- Resumable (skips splits where shards already have 26-class targets)

## Testing

1. Unit test for the remap function: 18-class input → 26-class output with correct indices and widening
2. Run script on real shards, verify with `create_webdataset_pipeline(datasets=["star-drums"])`
3. Verify multi-dataset loading: `datasets=["egmd_upload", "star-drums"]` produces uniform (26, 625) targets
