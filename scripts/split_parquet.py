"""Split parquet datasets into train/test by source audio (no leakage).

Two-pass approach to avoid loading everything into memory:
1. Scan all shards to collect unique sources
2. Assign sources to train/test, then re-read and write split shards
"""
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SEED = 42
TEST_FRACTION = 0.1
ROWS_PER_SHARD = 200


def split_dataset(dataset_path: str, source_col: str = "source_audio"):
    path = Path(dataset_path).expanduser()
    feature_dir = path / "features"
    files = sorted(feature_dir.glob("train-*.parquet"))

    if not files:
        print(f"No train-*.parquet files found in {feature_dir}")
        return

    # Pass 1: collect unique sources
    print(f"Pass 1: scanning {len(files)} shards for unique sources...")
    sources = set()
    total_rows = 0
    for f in files:
        meta = pq.ParquetFile(f)
        total_rows += meta.metadata.num_rows
        t = pq.read_table(f, columns=[source_col])
        for i in range(len(t)):
            sources.add(t.column(source_col)[i].as_py())

    sources = sorted(sources)
    random.seed(SEED)
    random.shuffle(sources)

    n_test = max(1, int(len(sources) * TEST_FRACTION))
    test_sources = set(sources[:n_test])

    print(f"Total rows: {total_rows:,}, Unique sources: {len(sources)}")
    print(f"Test sources: {n_test}, Train sources: {len(sources) - n_test}")

    # Pass 2: read each shard, split rows, write to new files
    print("Pass 2: splitting and writing shards...")
    train_buf = []
    test_buf = []
    train_shard = 0
    test_shard = 0

    # Backup dir for originals
    backup_dir = feature_dir / "_original"
    backup_dir.mkdir(exist_ok=True)

    for f in files:
        t = pq.read_table(f)
        src_data = t.column(source_col)

        train_idx = []
        test_idx = []
        for i in range(len(t)):
            if src_data[i].as_py() in test_sources:
                test_idx.append(i)
            else:
                train_idx.append(i)

        if train_idx:
            train_buf.append(t.take(train_idx))
        if test_idx:
            test_buf.append(t.take(test_idx))

        # Flush train buffer
        while train_buf and sum(len(b) for b in train_buf) >= ROWS_PER_SHARD:
            merged = pa.concat_tables(train_buf)
            chunk = merged.slice(0, ROWS_PER_SHARD)
            remainder = merged.slice(ROWS_PER_SHARD)
            out = feature_dir / f"train-{train_shard:05d}.parquet.new"
            pq.write_table(chunk, out)
            train_shard += 1
            train_buf = [remainder] if len(remainder) > 0 else []

        # Flush test buffer
        while test_buf and sum(len(b) for b in test_buf) >= ROWS_PER_SHARD:
            merged = pa.concat_tables(test_buf)
            chunk = merged.slice(0, ROWS_PER_SHARD)
            remainder = merged.slice(ROWS_PER_SHARD)
            out = feature_dir / f"test-{test_shard:05d}.parquet.new"
            pq.write_table(chunk, out)
            test_shard += 1
            test_buf = [remainder] if len(remainder) > 0 else []

        # Move original to backup
        f.rename(backup_dir / f.name)

    # Flush remaining
    if train_buf:
        merged = pa.concat_tables(train_buf)
        if len(merged) > 0:
            out = feature_dir / f"train-{train_shard:05d}.parquet.new"
            pq.write_table(merged, out)
            train_shard += 1

    if test_buf:
        merged = pa.concat_tables(test_buf)
        if len(merged) > 0:
            out = feature_dir / f"test-{test_shard:05d}.parquet.new"
            pq.write_table(merged, out)
            test_shard += 1

    # Rename .new files to final names
    for f in feature_dir.glob("*.parquet.new"):
        f.rename(f.with_suffix("").with_suffix(".parquet"))

    train_rows = sum(
        pq.ParquetFile(f).metadata.num_rows
        for f in feature_dir.glob("train-*.parquet")
    )
    test_rows = sum(
        pq.ParquetFile(f).metadata.num_rows
        for f in feature_dir.glob("test-*.parquet")
    )
    print(f"Wrote {train_shard} train shards ({train_rows:,} rows)")
    print(f"Wrote {test_shard} test shards ({test_rows:,} rows)")
    print(f"Originals backed up to {backup_dir}")
    print()


if __name__ == "__main__":
    for ds, col in [
        ("~/Documents/Datasets/parquet/e-gmd-aug", "source_audio"),
        ("~/Documents/Datasets/parquet/star-drums-aug", "source_audio"),
    ]:
        print(f"=== Splitting {ds} ===")
        split_dataset(ds, col)
