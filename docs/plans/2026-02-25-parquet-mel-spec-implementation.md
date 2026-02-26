# Parquet Mel Spectrogram Dataset — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pre-compute mel spectrograms and targets into an HF parquet dataset so training loads tensors instead of raw audio.

**Architecture:** A preprocessing script reuses existing dataset classes to compute mel specs + targets, stores them as flat float arrays in an HF `datasets.Dataset`, and pushes to Hub. A thin `ParquetDataset` wrapper reshapes rows back to tensors for the training loop. CLI gains `--dataset parquet` option.

**Tech Stack:** HF `datasets` library, existing `EGMDDataset`/`STARDataset`, PyTorch, `huggingface_hub`

---

### Task 1: Add `datasets` dependency

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add dependency**

Add `datasets` to the project dependencies in `pyproject.toml`:

```toml
dependencies = [
    "torch>=2.2",
    "torchaudio>=2.2",
    "pretty_midi>=0.2.10",
    "pyyaml>=6.0",
    "tqdm>=4.66",
    "mir_eval>=0.7",
    "datasets>=3.0",
]
```

**Step 2: Install**

Run: `pip install -e ".[dev]"`

**Step 3: Verify**

Run: `python -c "import datasets; print(datasets.__version__)"`
Expected: Version 3.x printed

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add HF datasets library"
```

---

### Task 2: Write `ParquetDataset` with tests (TDD)

**Files:**
- Create: `src/drumscribble/data/parquet.py`
- Create: `tests/test_parquet.py`

**Step 1: Write the failing tests**

Create `tests/test_parquet.py`:

```python
import torch
import pytest
from datasets import Dataset

from drumscribble.data.parquet import ParquetDataset
from drumscribble.config import NUM_CLASSES


@pytest.fixture
def fake_hf_dataset():
    """Create an in-memory HF dataset mimicking the parquet schema."""
    n_frames = 625
    n_mels = 128
    rows = {
        "mel": [torch.randn(1, n_mels, n_frames).flatten().tolist() for _ in range(10)],
        "onset_target": [torch.zeros(NUM_CLASSES, n_frames).flatten().tolist() for _ in range(10)],
        "vel_target": [torch.zeros(NUM_CLASSES, n_frames).flatten().tolist() for _ in range(10)],
        "source": ["egmd"] * 5 + ["star"] * 5,
    }
    return Dataset.from_dict(rows)


def test_parquet_dataset_len(fake_hf_dataset):
    ds = ParquetDataset(fake_hf_dataset)
    assert len(ds) == 10


def test_parquet_dataset_getitem_shapes(fake_hf_dataset):
    ds = ParquetDataset(fake_hf_dataset)
    mel, onset, vel = ds[0]
    assert mel.shape == (1, 128, 625)
    assert onset.shape == (NUM_CLASSES, 625)
    assert vel.shape == (NUM_CLASSES, 625)
    assert mel.dtype == torch.float32
    assert onset.dtype == torch.float32
    assert vel.dtype == torch.float32


def test_parquet_dataset_filter_source(fake_hf_dataset):
    ds = ParquetDataset(fake_hf_dataset, source="egmd")
    assert len(ds) == 5
    ds_star = ParquetDataset(fake_hf_dataset, source="star")
    assert len(ds_star) == 5


def test_parquet_dataset_no_filter(fake_hf_dataset):
    ds = ParquetDataset(fake_hf_dataset, source=None)
    assert len(ds) == 10


def test_parquet_dataset_works_with_dataloader(fake_hf_dataset):
    from torch.utils.data import DataLoader
    ds = ParquetDataset(fake_hf_dataset)
    loader = DataLoader(ds, batch_size=4)
    mel_batch, onset_batch, vel_batch = next(iter(loader))
    assert mel_batch.shape == (4, 1, 128, 625)
    assert onset_batch.shape == (4, NUM_CLASSES, 625)
    assert vel_batch.shape == (4, NUM_CLASSES, 625)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_parquet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'drumscribble.data.parquet'`

**Step 3: Write minimal implementation**

Create `src/drumscribble/data/parquet.py`:

```python
"""HF parquet dataset wrapper for pre-computed mel spectrograms."""
import torch
from torch.utils.data import Dataset as TorchDataset

from drumscribble.config import NUM_CLASSES, N_MELS


class ParquetDataset(TorchDataset):
    """Wraps an HF datasets.Dataset of pre-computed mel specs and targets.

    Each row contains flattened float arrays that get reshaped to tensors:
    - mel: (1, N_MELS, T)
    - onset_target: (NUM_CLASSES, T)
    - vel_target: (NUM_CLASSES, T)
    """

    def __init__(self, hf_dataset, source: str | None = None):
        if source is not None:
            hf_dataset = hf_dataset.filter(lambda row: row["source"] == source)
        self.dataset = hf_dataset
        # Infer frame count from first row
        mel_len = len(self.dataset[0]["mel"])
        self.n_frames = mel_len // N_MELS

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataset[idx]
        mel = torch.tensor(row["mel"], dtype=torch.float32).reshape(1, N_MELS, self.n_frames)
        onset = torch.tensor(row["onset_target"], dtype=torch.float32).reshape(NUM_CLASSES, self.n_frames)
        vel = torch.tensor(row["vel_target"], dtype=torch.float32).reshape(NUM_CLASSES, self.n_frames)
        return mel, onset, vel
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_parquet.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/drumscribble/data/parquet.py tests/test_parquet.py
git commit -m "feat: ParquetDataset wrapper for pre-computed mel specs"
```

---

### Task 3: Integrate `--dataset parquet` into CLI trainer

**Files:**
- Modify: `src/drumscribble/cli/train.py`
- Modify: `configs/train/default.yaml`

**Step 1: Update `build_dataset()` in CLI**

In `src/drumscribble/cli/train.py`, add parquet import and branch:

Add to imports:

```python
from drumscribble.data.parquet import ParquetDataset
```

Add `"parquet"` to the `--dataset` choices on line 91:

```python
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["egmd", "star", "multi", "parquet"],
        help="Dataset to use (overrides config)",
    )
```

Add two new CLI args after `--resume` (after line 97):

```python
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="HF Hub dataset repo for parquet mode (overrides config)",
    )
    parser.add_argument(
        "--parquet-source",
        type=str,
        default=None,
        choices=["egmd", "star"],
        help="Filter parquet dataset by source",
    )
```

Add a `parquet` branch to `build_dataset()` function (after the `multi` branch, before `else`). This requires passing extra args, so change the signature and add a branch:

```python
def build_dataset(dataset_name: str, data_cfg: dict, train_cfg: dict,
                  hf_dataset: str | None = None, parquet_source: str | None = None):
```

Add before the `else` raise:

```python
    elif dataset_name == "parquet":
        from datasets import load_dataset
        repo = hf_dataset or data_cfg.get("hf_dataset_repo", "zkeown/drumscribble-mel-specs")
        hf_ds = load_dataset(repo, split="train")
        return ParquetDataset(hf_ds, source=parquet_source)
```

Update the two call sites in `main()` to pass the new args:

```python
    # In the multi branch (~line 138):
    datasets = build_dataset("multi", data_cfg, train_cfg)
    # becomes:
    datasets = build_dataset("multi", data_cfg, train_cfg,
                             hf_dataset=args.hf_dataset, parquet_source=args.parquet_source)

    # In the single-dataset branch (~line 161):
    dataset = build_dataset(dataset_name, data_cfg, train_cfg)
    # becomes:
    dataset = build_dataset(dataset_name, data_cfg, train_cfg,
                            hf_dataset=args.hf_dataset, parquet_source=args.parquet_source)
```

**Step 2: Update config**

Add to `configs/train/default.yaml` under `data:`:

```yaml
data:
  egmd_root: ~/Documents/Datasets/e-gmd
  star_root: ~/Documents/Datasets/star-drums
  hf_dataset_repo: zkeown/drumscribble-mel-specs
```

**Step 3: Verify existing tests still pass**

Run: `pytest tests/ -v --ignore=tests/test_egmd.py --ignore=tests/test_star.py -x`
Expected: All tests PASS (egmd/star tests may fail without raw data, so skip them)

**Step 4: Commit**

```bash
git add src/drumscribble/cli/train.py configs/train/default.yaml
git commit -m "feat: add --dataset parquet mode to CLI trainer"
```

---

### Task 4: Update HF Jobs training script

**Files:**
- Modify: `scripts/train_hf_job.py`

**Step 1: Add parquet support**

Add `"parquet"` to the `--dataset` choices (line 76):

```python
    parser.add_argument("--dataset", type=str, default="parquet",
                        choices=["egmd", "star", "multi", "parquet"],
                        help="Dataset to train on")
```

Note: default changes from `"egmd"` to `"parquet"` since parquet is the primary path now.

Add two new CLI args after `--dataset-weights` (after line 88):

```python
    parser.add_argument("--hf-dataset", type=str, default="zkeown/drumscribble-mel-specs",
                        help="HF Hub dataset repo for parquet mode")
    parser.add_argument("--parquet-source", type=str, default=None,
                        choices=["egmd", "star"],
                        help="Filter parquet dataset by source")
```

Add `"datasets>=3.0"` to the PEP 723 dependencies block at the top (line 5):

```python
# dependencies = [
#     "drumscribble @ git+https://github.com/zakkeown/drumscribble.git",
#     "huggingface_hub[hf_xet]",
#     "pyyaml",
#     "datasets>=3.0",
# ]
```

Add a parquet branch in the dataset loading section (after the `multi` elif, before the optimizer section ~line 208):

```python
    elif args.dataset == "parquet":
        from datasets import load_dataset as hf_load_dataset
        from drumscribble.data.parquet import ParquetDataset

        print(f"Loading parquet dataset from {args.hf_dataset}...")
        hf_ds = hf_load_dataset(args.hf_dataset, split="train")
        dataset = ParquetDataset(hf_ds, source=args.parquet_source)
        src_label = f" (source={args.parquet_source})" if args.parquet_source else ""
        print(f"Parquet training samples{src_label}: {len(dataset):,}")
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True, collate_fn=collate_fn,
        )
```

**Step 2: Skip raw data download when using parquet**

Wrap the existing download logic in a condition — the existing `if args.dataset in ("egmd", "multi"):` and `if args.dataset in ("star", "multi"):` blocks already handle this correctly since `"parquet"` won't match those conditions.

**Step 3: Commit**

```bash
git add scripts/train_hf_job.py
git commit -m "feat: add --dataset parquet to HF Jobs training script"
```

---

### Task 5: Build the preprocessing script

**Files:**
- Create: `scripts/build_parquet_dataset.py`

**Step 1: Write the script**

Create `scripts/build_parquet_dataset.py`:

```python
#!/usr/bin/env python3
"""Build parquet mel spectrogram dataset from raw E-GMD and STAR.

Downloads raw datasets, computes mel spectrograms and targets using
existing dataset classes, and pushes to HF Hub as a parquet dataset.

Usage:
    python scripts/build_parquet_dataset.py --output-repo zkeown/drumscribble-mel-specs

    # Skip download if you already have raw data:
    python scripts/build_parquet_dataset.py \
        --egmd-root ~/Documents/Datasets/e-gmd \
        --star-root ~/Documents/Datasets/star-drums \
        --skip-download
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from tqdm import tqdm


EGMD_URL = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip"
EGMD_SHA256 = "7d9a264fb4c9eabd9fec09d5f8e333192f529b1a1b845d170279a977ac436053"

STAR_ZENODO_RECORD = "15690078"
STAR_PARTS = [
    "STAR_Drums_full.zip.part-aa",
    "STAR_Drums_full.zip.part-ab",
    "STAR_Drums_full.zip.part-ac",
    "STAR_Drums_full.zip.part-ad",
    "STAR_Drums_full.zip.part-ae",
    "STAR_Drums_full.zip.part-af",
]


def download_egmd(dest: Path) -> Path:
    """Download and extract E-GMD dataset."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "e-gmd-v1.0.0.zip"

    if not zip_path.exists():
        print(f"Downloading E-GMD ({EGMD_URL})...")
        subprocess.run(["curl", "-L", "-o", str(zip_path), EGMD_URL], check=True)
    else:
        print(f"E-GMD zip already exists at {zip_path}")

    egmd_root = dest / "e-gmd-v1.0.0"
    if not egmd_root.exists():
        print("Extracting E-GMD...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(dest)], check=True)

    return egmd_root


def download_star(dest: Path) -> Path:
    """Download and extract STAR Drums dataset from Zenodo."""
    dest.mkdir(parents=True, exist_ok=True)

    for part in STAR_PARTS:
        part_path = dest / part
        if not part_path.exists():
            url = f"https://zenodo.org/records/{STAR_ZENODO_RECORD}/files/{part}"
            print(f"Downloading {part}...")
            subprocess.run(["curl", "-L", "-o", str(part_path), url], check=True)
        else:
            print(f"{part} already exists")

    combined = dest / "STAR_Drums_full.zip"
    if not combined.exists():
        print("Combining STAR parts...")
        subprocess.run(
            f"cat {dest}/STAR_Drums_full.zip.part-* > {combined}",
            shell=True, check=True,
        )

    star_root = dest / "STAR_Drums"
    if not star_root.exists():
        print("Extracting STAR...")
        subprocess.run(["unzip", "-q", str(combined), "-d", str(dest)], check=True)

    return star_root


def process_dataset(dataset, source: str, split: str) -> dict:
    """Extract all chunks from a dataset into flat row dicts."""
    rows = {"mel": [], "onset_target": [], "vel_target": [], "source": []}
    print(f"Processing {source}/{split}: {len(dataset)} chunks...")

    for idx in tqdm(range(len(dataset)), desc=f"{source}/{split}"):
        mel, onset, vel = dataset[idx]
        rows["mel"].append(mel.flatten().tolist())
        rows["onset_target"].append(onset.flatten().tolist())
        rows["vel_target"].append(vel.flatten().tolist())
        rows["source"].append(source)

    return rows


def merge_rows(*row_dicts: dict) -> dict:
    """Merge multiple row dicts into one."""
    merged = {"mel": [], "onset_target": [], "vel_target": [], "source": []}
    for rd in row_dicts:
        for key in merged:
            merged[key].extend(rd[key])
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build parquet mel spec dataset")
    parser.add_argument("--output-repo", type=str, default="zkeown/drumscribble-mel-specs")
    parser.add_argument("--egmd-root", type=str, default=None,
                        help="Path to existing E-GMD data (skip download)")
    parser.add_argument("--star-root", type=str, default=None,
                        help="Path to existing STAR data (skip download)")
    parser.add_argument("--download-dir", type=str, default="~/Documents/Datasets",
                        help="Directory to download raw datasets into")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use --egmd-root and --star-root")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--private", action="store_true",
                        help="Make the HF dataset repo private")
    args = parser.parse_args()

    download_dir = Path(args.download_dir).expanduser()

    # --- Get raw data ---
    if args.skip_download:
        egmd_root = Path(args.egmd_root).expanduser() if args.egmd_root else None
        star_root = Path(args.star_root).expanduser() if args.star_root else None
    else:
        egmd_root = download_egmd(download_dir / "e-gmd")
        star_root = download_star(download_dir / "star-drums")

    # --- Import dataset classes ---
    from drumscribble.data.egmd import EGMDDataset
    from drumscribble.data.star import STARDataset

    # --- Process E-GMD ---
    train_rows = []
    val_rows = []

    if egmd_root:
        print(f"\n=== E-GMD from {egmd_root} ===")
        egmd_train = EGMDDataset(egmd_root, split="train", chunk_seconds=args.chunk_seconds)
        egmd_val = EGMDDataset(egmd_root, split="validation", chunk_seconds=args.chunk_seconds)
        train_rows.append(process_dataset(egmd_train, "egmd", "train"))
        val_rows.append(process_dataset(egmd_val, "egmd", "validation"))

    # --- Process STAR ---
    if star_root:
        print(f"\n=== STAR from {star_root} ===")
        star_train = STARDataset(star_root, split="training", chunk_seconds=args.chunk_seconds)
        star_val = STARDataset(star_root, split="validation", chunk_seconds=args.chunk_seconds)
        train_rows.append(process_dataset(star_train, "star", "train"))
        val_rows.append(process_dataset(star_val, "star", "validation"))

    if not train_rows:
        print("ERROR: No datasets processed. Provide --egmd-root or --star-root, or allow downloads.")
        sys.exit(1)

    # --- Build HF DatasetDict ---
    train_merged = merge_rows(*train_rows)
    val_merged = merge_rows(*val_rows) if val_rows else None

    print(f"\nTrain rows: {len(train_merged['mel']):,}")
    splits = {"train": Dataset.from_dict(train_merged)}

    if val_merged and val_merged["mel"]:
        print(f"Validation rows: {len(val_merged['mel']):,}")
        splits["validation"] = Dataset.from_dict(val_merged)

    dataset_dict = DatasetDict(splits)

    # --- Push to Hub ---
    print(f"\nPushing to {args.output_repo}...")
    dataset_dict.push_to_hub(args.output_repo, private=args.private)
    print(f"Done! Dataset available at https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify script parses correctly**

Run: `python scripts/build_parquet_dataset.py --help`
Expected: Help text printed with all args

**Step 3: Commit**

```bash
git add scripts/build_parquet_dataset.py
git commit -m "feat: preprocessing script to build parquet mel spec dataset"
```

---

### Task 6: End-to-end integration test

**Files:**
- Modify: `tests/test_parquet.py`

**Step 1: Add integration test with training loop**

Append to `tests/test_parquet.py`:

```python
def test_parquet_dataset_with_training_loop(fake_hf_dataset):
    """Verify ParquetDataset works end-to-end with the training loop."""
    from torch.utils.data import DataLoader
    from drumscribble.model.drumscribble import DrumscribbleCNN
    from drumscribble.loss import DrumscribbleLoss
    from drumscribble.train import train_one_epoch, create_optimizer

    ds = ParquetDataset(fake_hf_dataset)
    loader = DataLoader(ds, batch_size=4, drop_last=True)

    model = DrumscribbleCNN(
        backbone_dims=(32, 32, 32, 32),
        backbone_depths=(1, 1, 1, 1),
        num_attn_layers=1,
    )
    optimizer = create_optimizer(model, lr=1e-3)
    loss_fn = DrumscribbleLoss()

    avg_loss = train_one_epoch(model, loader, optimizer, loss_fn, device="cpu")
    assert avg_loss > 0
    assert not torch.isnan(torch.tensor(avg_loss))
```

**Step 2: Run all parquet tests**

Run: `pytest tests/test_parquet.py -v`
Expected: All 6 tests PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/test_egmd.py --ignore=tests/test_star.py -x`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_parquet.py
git commit -m "test: add end-to-end integration test for parquet training"
```

---

### Task 7: Download datasets and build parquet

**This is the actual data processing step — run only after all code is committed and tested.**

**Step 1: Run the build script**

Run: `python scripts/build_parquet_dataset.py --output-repo zkeown/drumscribble-mel-specs`

This will:
1. Download E-GMD (~90GB) to `~/Documents/Datasets/e-gmd/`
2. Download STAR (~181GB) to `~/Documents/Datasets/star-drums/`
3. Process all chunks into mel specs + targets
4. Push parquet dataset to HF Hub

Estimated time: Several hours (mostly download time).

**Step 2: Verify the dataset on HF Hub**

Run: `python -c "from datasets import load_dataset; ds = load_dataset('zkeown/drumscribble-mel-specs', split='train'); print(f'Rows: {len(ds):,}'); print(ds[0].keys())"`
Expected: Row count printed, keys are `mel`, `onset_target`, `vel_target`, `source`

**Step 3: Test local training with parquet**

Run: `python -m drumscribble.cli.train --dataset parquet --epochs 1 --device cpu`
Expected: One epoch completes, loss printed

---

### Task 8: Save memory note

**Step 1: Save to auto memory**

Record the key decisions and paths in the auto memory file so future sessions have context about the parquet training setup.
