# Remap STAR Shards Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remap STAR feature shards from 18-class STAR taxonomy to 26-class GM taxonomy with onset widening, so STAR and E-GMD shards can be mixed in the WebDataset training pipeline.

**Architecture:** A pure numpy remapping script reads existing STAR feature tar shards, maps 18-class onset/velocity arrays to 26-class GM arrays using `STAR_ABBREV_TO_GM` + `GM_NOTE_TO_INDEX`, applies `TARGET_WIDENING` to onsets, and writes new shards in-place. A reusable `remap_star_targets()` function lives in the drumscribble package for testability.

**Tech Stack:** numpy, tarfile (stdlib), drumscribble.config mappings

---

### Task 1: Add `STAR_CLASSES` constant and `remap_star_targets()` function

The STAR class ordering is defined in `~/Documents/Datasets/scripts/star_annotation_to_targets.py` but not in drumscribble. We need it here to know which row index maps to which abbreviation.

**Files:**
- Modify: `src/drumscribble/config.py`
- Create: `src/drumscribble/data/remap.py`
- Test: `tests/test_remap.py`

**Step 1: Add `STAR_CLASSES` to config.py**

Add after the `STAR_ABBREV_TO_GM` dict (line 40 in `src/drumscribble/config.py`):

```python
# STAR 18-class ordering (matches annotation_to_targets index convention)
STAR_CLASSES = [
    "BD", "SD", "CHH", "OHH", "PHH", "HT", "MT", "LT",
    "CRC", "SPC", "CHC", "RD", "RB", "CB", "CL", "CLP", "SS", "TB",
]
```

**Step 2: Write the failing test**

Create `tests/test_remap.py`:

```python
"""Tests for STAR 18-class to GM 26-class target remapping."""
import numpy as np
import pytest

from drumscribble.config import (
    GM_NOTE_TO_INDEX, NUM_CLASSES, STAR_ABBREV_TO_GM, STAR_CLASSES,
    TARGET_WIDENING,
)


def test_remap_star_targets_shape():
    """18-class input produces 26-class output."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.zeros((18, 100), dtype=np.float32)
    vel_18 = np.zeros((18, 100), dtype=np.float32)

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    assert onset_26.shape == (26, 100)
    assert vel_26.shape == (26, 100)
    assert onset_26.dtype == np.float32
    assert vel_26.dtype == np.float32


def test_remap_star_targets_mapping():
    """Bass drum (STAR index 0) maps to GM note 36 (GM index 1)."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.zeros((18, 50), dtype=np.float32)
    vel_18 = np.zeros((18, 50), dtype=np.float32)

    # BD is STAR index 0, set onset at frame 25
    onset_18[0, 25] = 1.0
    vel_18[0, 25] = 0.8

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    # BD -> GM note 36 -> GM index 1
    gm_idx = GM_NOTE_TO_INDEX[36]
    assert gm_idx == 1

    # Onset should have widening pattern centered at frame 25
    assert onset_26[gm_idx, 25] == pytest.approx(1.0)
    assert onset_26[gm_idx, 24] == pytest.approx(0.6)
    assert onset_26[gm_idx, 23] == pytest.approx(0.3)
    assert onset_26[gm_idx, 26] == pytest.approx(0.6)
    assert onset_26[gm_idx, 27] == pytest.approx(0.3)

    # Velocity only at center frame
    assert vel_26[gm_idx, 25] == pytest.approx(0.8)
    assert vel_26[gm_idx, 24] == 0.0  # no velocity on widened frames


def test_remap_star_targets_all_classes_mapped():
    """Every STAR class maps to a valid GM index."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.zeros((18, 20), dtype=np.float32)
    vel_18 = np.zeros((18, 20), dtype=np.float32)

    # Set all 18 classes active at frame 10
    for i in range(18):
        onset_18[i, 10] = 1.0
        vel_18[i, 10] = 0.5

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    # Check each STAR class landed in the right GM row
    for star_idx, abbrev in enumerate(STAR_CLASSES):
        gm_note = STAR_ABBREV_TO_GM[abbrev]
        gm_idx = GM_NOTE_TO_INDEX[gm_note]
        assert onset_26[gm_idx, 10] == pytest.approx(1.0), f"{abbrev} missing at GM idx {gm_idx}"
        assert vel_26[gm_idx, 10] == pytest.approx(0.5), f"{abbrev} vel missing"


def test_remap_star_targets_unmapped_rows_zero():
    """GM classes with no STAR equivalent stay zero."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.ones((18, 10), dtype=np.float32)
    vel_18 = np.ones((18, 10), dtype=np.float32)

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    # Find GM indices NOT covered by any STAR class
    mapped_gm_indices = set()
    for abbrev in STAR_CLASSES:
        gm_note = STAR_ABBREV_TO_GM[abbrev]
        mapped_gm_indices.add(GM_NOTE_TO_INDEX[gm_note])

    for gm_idx in range(NUM_CLASSES):
        if gm_idx not in mapped_gm_indices:
            assert onset_26[gm_idx].sum() == 0.0, f"GM idx {gm_idx} should be zero"
            assert vel_26[gm_idx].sum() == 0.0


def test_remap_widening_clamps_to_one():
    """Overlapping widened onsets don't exceed 1.0."""
    from drumscribble.data.remap import remap_star_targets

    onset_18 = np.zeros((18, 10), dtype=np.float32)
    vel_18 = np.zeros((18, 10), dtype=np.float32)

    # Two BD onsets 2 frames apart — widening will overlap
    onset_18[0, 3] = 1.0
    onset_18[0, 5] = 1.0

    onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

    gm_idx = GM_NOTE_TO_INDEX[36]  # BD
    assert onset_26[gm_idx].max() <= 1.0
```

**Step 3: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_remap.py -v`
Expected: FAIL — `ImportError: cannot import name 'STAR_CLASSES'` or `ModuleNotFoundError: No module named 'drumscribble.data.remap'`

**Step 4: Implement `remap_star_targets()`**

Create `src/drumscribble/data/remap.py`:

```python
"""Remap STAR 18-class targets to 26-class GM taxonomy."""
import numpy as np

from drumscribble.config import (
    GM_NOTE_TO_INDEX,
    NUM_CLASSES,
    STAR_ABBREV_TO_GM,
    STAR_CLASSES,
    TARGET_WIDENING,
)

# Precompute STAR index -> GM index mapping
_STAR_TO_GM_INDEX = [
    GM_NOTE_TO_INDEX[STAR_ABBREV_TO_GM[abbrev]]
    for abbrev in STAR_CLASSES
]


def remap_star_targets(
    onset_18: np.ndarray,
    vel_18: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap (18, T) STAR targets to (26, T) GM targets with onset widening.

    Args:
        onset_18: Binary onset targets, shape (18, T).
        vel_18: Velocity targets normalized to [0, 1], shape (18, T).

    Returns:
        Tuple of (onset_26, vel_26), each shape (26, T), float32.
    """
    n_frames = onset_18.shape[1]
    onset_26 = np.zeros((NUM_CLASSES, n_frames), dtype=np.float32)
    vel_26 = np.zeros((NUM_CLASSES, n_frames), dtype=np.float32)

    half_w = len(TARGET_WIDENING) // 2

    for star_idx, gm_idx in enumerate(_STAR_TO_GM_INDEX):
        # Copy velocity directly (no widening)
        vel_26[gm_idx] = vel_18[star_idx]

        # Apply widening to onsets
        active_frames = np.nonzero(onset_18[star_idx])[0]
        for center in active_frames:
            for i, w in enumerate(TARGET_WIDENING):
                frame = center - half_w + i
                if 0 <= frame < n_frames:
                    onset_26[gm_idx, frame] = max(onset_26[gm_idx, frame], w)

    return onset_26, vel_26
```

**Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_remap.py -v`
Expected: 5 passed

**Step 6: Commit**

```bash
git add src/drumscribble/config.py src/drumscribble/data/remap.py tests/test_remap.py
git commit -m "feat: add STAR 18-class to GM 26-class target remapping"
```

---

### Task 2: Write the shard remapping script

**Files:**
- Create: `scripts/remap_star_shards.py`

**Step 1: Write the script**

```python
"""Remap STAR feature shards from 18-class to 26-class GM taxonomy.

Reads existing feature shards, remaps onset/velocity targets using
drumscribble's STAR-to-GM mapping with onset widening, and writes
new shards in-place.

Usage:
    uv run python scripts/remap_star_shards.py \\
        --shard-root ~/Documents/Datasets/build/star-drums
"""
import argparse
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from drumscribble.config import NUM_CLASSES
from drumscribble.data.remap import remap_star_targets


def remap_shard(src_path: Path, dst_path: Path) -> int:
    """Remap one feature shard tar file.

    Returns the number of samples processed.
    """
    count = 0
    with tarfile.open(src_path, "r") as src, tarfile.open(dst_path, "w") as dst:
        members = src.getmembers()

        # Group members by sample key
        samples: dict[str, dict[str, tarfile.TarInfo]] = {}
        for m in members:
            # Split on first dot to get sample key
            dot_idx = m.name.index(".")
            key = m.name[:dot_idx]
            suffix = m.name[dot_idx + 1:]
            if key not in samples:
                samples[key] = {}
            samples[key][suffix] = m

        for key, parts in samples.items():
            # Read existing arrays
            onset_f = src.extractfile(parts["onset_targets.npy"])
            onset_18 = np.load(io.BytesIO(onset_f.read()))

            vel_f = src.extractfile(parts["velocity_targets.npy"])
            vel_18 = np.load(io.BytesIO(vel_f.read()))

            # Skip if already remapped
            if onset_18.shape[0] == NUM_CLASSES:
                # Copy all members unchanged
                for suffix, member in parts.items():
                    f = src.extractfile(member)
                    dst.addfile(member, f)
                count += 1
                continue

            # Remap
            onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

            # Write mel unchanged
            mel_member = parts["mel_spectrogram.npy"]
            mel_f = src.extractfile(mel_member)
            dst.addfile(mel_member, mel_f)

            # Write remapped onset
            buf = io.BytesIO()
            np.save(buf, onset_26)
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{key}.onset_targets.npy")
            info.size = buf.getbuffer().nbytes
            dst.addfile(info, buf)

            # Write remapped velocity
            buf = io.BytesIO()
            np.save(buf, vel_26)
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{key}.velocity_targets.npy")
            info.size = buf.getbuffer().nbytes
            dst.addfile(info, buf)

            # Write updated params
            params_f = src.extractfile(parts["params.json"])
            params = json.loads(params_f.read())
            params["n_classes"] = NUM_CLASSES
            params_bytes = json.dumps(params).encode()
            info = tarfile.TarInfo(name=f"{key}.params.json")
            info.size = len(params_bytes)
            dst.addfile(info, io.BytesIO(params_bytes))

            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Remap STAR feature shards from 18-class to 26-class GM"
    )
    parser.add_argument(
        "--shard-root",
        type=str,
        required=True,
        help="Root of STAR dataset (e.g. ~/Documents/Datasets/build/star-drums)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying files",
    )
    args = parser.parse_args()

    root = Path(args.shard_root).expanduser()
    features_dir = root / "data" / "features"

    for split in ["train", "validation", "test"]:
        split_dir = features_dir / split
        if not split_dir.exists():
            print(f"Skipping {split}: {split_dir} not found")
            continue

        shards = sorted(split_dir.glob("feature-shard-*.tar"))
        if not shards:
            print(f"Skipping {split}: no shards found")
            continue

        print(f"{split}: {len(shards)} shards")

        if args.dry_run:
            continue

        # Write to temp dir, then swap
        tmp_dir = Path(tempfile.mkdtemp(dir=split_dir.parent))
        try:
            for shard_path in shards:
                dst_path = tmp_dir / shard_path.name
                count = remap_shard(shard_path, dst_path)
                print(f"  {shard_path.name}: {count} samples remapped")

            # Verify all output shards exist and have content
            for shard_path in shards:
                dst_path = tmp_dir / shard_path.name
                assert dst_path.exists(), f"Missing output: {dst_path}"
                assert dst_path.stat().st_size > 0, f"Empty output: {dst_path}"

            # Swap: move originals out, move new in
            backup_dir = Path(tempfile.mkdtemp(dir=split_dir.parent))
            for shard_path in shards:
                shutil.move(str(shard_path), str(backup_dir / shard_path.name))
            for shard_path in shards:
                shutil.move(str(tmp_dir / shard_path.name), str(shard_path))

            # Clean up backup
            shutil.rmtree(backup_dir)
            print(f"  {split} done!")

        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    print("All splits remapped.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/remap_star_shards.py
git commit -m "feat: add script to remap STAR shards to 26-class GM"
```

---

### Task 3: Run the remapping on real shards

**Step 1: Dry run to verify**

Run: `uv run python scripts/remap_star_shards.py --shard-root ~/Documents/Datasets/build/star-drums --dry-run`
Expected: Lists 8 train, 2 validation, 1 test shard without modifying anything.

**Step 2: Run the remapping**

Run: `uv run python scripts/remap_star_shards.py --shard-root ~/Documents/Datasets/build/star-drums`
Expected: All 11 shards remapped successfully.

**Step 3: Verify output shapes**

```bash
uv run python -c "
import tarfile, numpy as np, io, json
from pathlib import Path

tar_path = next(Path('~/Documents/Datasets/build/star-drums/data/features/train/').expanduser().glob('*.tar'))
with tarfile.open(tar_path) as tar:
    for m in tar.getmembers()[:8]:
        if m.name.endswith('params.json'):
            f = tar.extractfile(m)
            params = json.loads(f.read())
            print('params:', params)
            assert params['n_classes'] == 26
        elif m.name.endswith('.npy'):
            f = tar.extractfile(m)
            arr = np.load(io.BytesIO(f.read()))
            print(f'{m.name.split(\".\", 1)[1]}: shape={arr.shape}')
print('Verification passed!')
"
```

Expected: `onset_targets.npy: shape=(26, T)`, `velocity_targets.npy: shape=(26, T)`, `n_classes: 26`.

---

### Task 4: Verify WebDataset pipeline loads STAR shards

**Step 1: Load STAR shards through the pipeline**

```bash
uv run python -c "
from drumscribble.data.webdataset_loader import create_webdataset_pipeline
from drumscribble.config import N_MELS, NUM_CLASSES

pipeline = create_webdataset_pipeline(
    shard_root='~/Documents/Datasets/build',
    datasets=['star-drums'],
    split='train',
    shuffle=False,
    epoch_size=10,
)
samples = list(pipeline)
print(f'Got {len(samples)} STAR samples')
mel, onset, vel = samples[0]
print(f'mel: {mel.shape}, onset: {onset.shape}, vel: {vel.shape}')
assert mel.shape == (N_MELS, 625)
assert onset.shape == (NUM_CLASSES, 625)
assert vel.shape == (NUM_CLASSES, 625)
print('STAR pipeline OK!')
"
```

**Step 2: Load both datasets together**

```bash
uv run python -c "
from drumscribble.data.webdataset_loader import create_webdataset_pipeline
from drumscribble.config import N_MELS, NUM_CLASSES

pipeline = create_webdataset_pipeline(
    shard_root='~/Documents/Datasets/build',
    datasets=['egmd_upload', 'star-drums'],
    split='train',
    shuffle=False,
    epoch_size=20,
)
samples = list(pipeline)
print(f'Got {len(samples)} mixed samples')
for i, (mel, onset, vel) in enumerate(samples):
    assert mel.shape == (N_MELS, 625), f'Sample {i}: mel shape {mel.shape}'
    assert onset.shape == (NUM_CLASSES, 625), f'Sample {i}: onset shape {onset.shape}'
print('Multi-dataset pipeline OK!')
"
```

**Step 3: Update default config**

Modify `configs/train/default.yaml` to include star-drums:

```yaml
data:
  shard_root: ~/Documents/Datasets/build
  datasets:
    - egmd_upload
    - star-drums
  shuffle_buffer: 5000
  estimated_samples: 41000  # ~35k E-GMD + ~6k STAR
```

**Step 4: Commit**

```bash
git add configs/train/default.yaml
git commit -m "feat: enable STAR dataset in training config"
```
