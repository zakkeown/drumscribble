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
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict, concatenate_datasets
from tqdm import tqdm

NUM_WORKERS = max(1, os.cpu_count() - 2)  # leave 2 cores free


EGMD_URL = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip"

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
        with open(combined, "wb") as out:
            for part in sorted(dest.glob("STAR_Drums_full.zip.part-*")):
                with open(part, "rb") as f:
                    while chunk := f.read(8192):
                        out.write(chunk)

    star_root = dest / "STAR_Drums"
    if not star_root.exists():
        print("Extracting STAR...")
        subprocess.run(["unzip", "-q", str(combined), "-d", str(dest)], check=True)

    return star_root


def process_dataset(dataset, source: str, split: str, num_workers: int = NUM_WORKERS) -> Dataset | None:
    """Process a PyTorch dataset into an HF Dataset, streaming to disk via from_generator."""
    total = len(dataset)
    print(f"Processing {source}/{split}: {total} chunks with {num_workers} workers...")

    if total == 0:
        print(f"  {source}/{split}: skipped (0 chunks)")
        return None

    def gen():
        loader = DataLoader(
            dataset,
            batch_size=num_workers * 4,
            num_workers=num_workers,
            shuffle=False,
            prefetch_factor=4,
        )
        for mel_batch, onset_batch, vel_batch in tqdm(loader, desc=f"{source}/{split}"):
            b = mel_batch.shape[0]
            mel_np = mel_batch.reshape(b, -1).numpy()
            onset_np = onset_batch.reshape(b, -1).numpy()
            vel_np = vel_batch.reshape(b, -1).numpy()
            for i in range(b):
                yield {
                    "mel": mel_np[i],
                    "onset_target": onset_np[i],
                    "vel_target": vel_np[i],
                    "source": source,
                }

    ds = Dataset.from_generator(gen)
    print(f"  {source}/{split}: {len(ds):,} rows written to cache")
    return ds


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
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help=f"Number of DataLoader workers (default: {NUM_WORKERS})")
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

    # --- Process datasets into HF Datasets (streamed to disk) ---
    train_parts: list[Dataset] = []
    val_parts: list[Dataset] = []

    if egmd_root:
        print(f"\n=== E-GMD from {egmd_root} ===")
        egmd_train = EGMDDataset(egmd_root, split="train", chunk_seconds=args.chunk_seconds)
        egmd_val = EGMDDataset(egmd_root, split="validation", chunk_seconds=args.chunk_seconds)
        if ds := process_dataset(egmd_train, "egmd", "train", args.num_workers):
            train_parts.append(ds)
        if ds := process_dataset(egmd_val, "egmd", "validation", args.num_workers):
            val_parts.append(ds)

    if star_root:
        print(f"\n=== STAR from {star_root} ===")
        star_train = STARDataset(star_root, split="training", chunk_seconds=args.chunk_seconds)
        star_val = STARDataset(star_root, split="validation", chunk_seconds=args.chunk_seconds)
        if ds := process_dataset(star_train, "star", "train", args.num_workers):
            train_parts.append(ds)
        if ds := process_dataset(star_val, "star", "validation", args.num_workers):
            val_parts.append(ds)

    if not train_parts:
        print("ERROR: No datasets processed. Provide --egmd-root or --star-root, or allow downloads.")
        sys.exit(1)

    # --- Build HF DatasetDict ---
    train_ds = concatenate_datasets(train_parts) if len(train_parts) > 1 else train_parts[0]
    print(f"\nTrain rows: {len(train_ds):,}")
    splits = {"train": train_ds}

    if val_parts:
        val_ds = concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]
        print(f"Validation rows: {len(val_ds):,}")
        splits["validation"] = val_ds

    dataset_dict = DatasetDict(splits)

    # --- Push to Hub ---
    print(f"\nPushing to {args.output_repo}...")
    dataset_dict.push_to_hub(args.output_repo, private=args.private)
    print(f"Done! Dataset available at https://huggingface.co/datasets/{args.output_repo}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
