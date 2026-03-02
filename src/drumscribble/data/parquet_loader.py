"""Parquet-based loader for pre-computed feature datasets from HF Hub."""
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info

from drumscribble.config import FPS
from drumscribble.data.webdataset_loader import _chunk_sample, _to_tensors


def discover_parquet_shards(
    data_root: str | Path,
    datasets: list[str],
    split: str,
) -> list[str]:
    """Find parquet shard files under data_root.

    Expects directory structure from ``huggingface-cli download``:
        {data_root}/{dataset}/features/{split}-XXXXX.parquet

    Args:
        data_root: Root directory containing dataset subdirectories.
        datasets: List of dataset directory names.
        split: Data split (e.g. "train").

    Returns:
        Sorted list of absolute parquet file paths as strings.

    Raises:
        FileNotFoundError: If a dataset's features directory doesn't exist.
    """
    root = Path(data_root).expanduser()
    all_shards: list[str] = []
    for name in datasets:
        feature_dir = root / name / "features"
        if not feature_dir.exists():
            raise FileNotFoundError(
                f"Parquet feature directory not found: {feature_dir}"
            )
        shards = sorted(feature_dir.glob(f"{split}-*.parquet"))
        all_shards.extend(str(s) for s in shards)
    return all_shards


def _decode_row(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deserialize binary columns from a parquet row.

    Returns (mel, onset, vel) numpy arrays with correct shapes.
    """
    n_mels = row["n_mels"]
    n_frames = row["n_frames"]
    n_classes = row["n_classes"]

    mel = np.frombuffer(row["mel_spectrogram"], dtype=np.float32).reshape(
        n_mels, n_frames
    )
    onset = np.frombuffer(row["onset_targets"], dtype=np.float32).reshape(
        n_classes, n_frames
    )
    vel = np.frombuffer(row["velocity_targets"], dtype=np.float32).reshape(
        n_classes, n_frames
    )
    return mel, onset, vel


class _ShuffleBuffer:
    """Simple ring-buffer shuffle over an iterator.

    Fills a buffer of ``buffer_size`` elements, then yields a random
    element while refilling from the source iterator.
    """

    def __init__(self, iterator, buffer_size: int):
        self._iterator = iterator
        self._buffer_size = buffer_size

    def __iter__(self):
        buf: list = []
        it = iter(self._iterator)

        # Fill buffer
        for item in it:
            buf.append(item)
            if len(buf) >= self._buffer_size:
                break

        # Yield random, refill
        for item in it:
            idx = random.randint(0, len(buf) - 1)
            yield buf[idx]
            buf[idx] = item

        # Drain remaining
        random.shuffle(buf)
        yield from buf


class ParquetFeatureDataset(IterableDataset):
    """IterableDataset reading pre-computed features from parquet shards.

    Reads parquet files sequentially, decodes binary columns into numpy
    arrays, chunks variable-length samples into fixed-size segments,
    optionally shuffles via a ring buffer, and converts to tensors.
    """

    def __init__(
        self,
        shard_paths: list[str],
        chunk_frames: int = 625,
        shuffle: bool = True,
        shuffle_buffer: int = 5000,
    ):
        self.shard_paths = shard_paths
        self.chunk_frames = chunk_frames
        self.shuffle = shuffle
        self.shuffle_buffer = shuffle_buffer

    def _worker_shards(self) -> list[str]:
        """Partition shards across DataLoader workers."""
        info = get_worker_info()
        if info is None:
            return self.shard_paths
        per_worker = len(self.shard_paths) // info.num_workers
        remainder = len(self.shard_paths) % info.num_workers
        start = info.id * per_worker + min(info.id, remainder)
        end = start + per_worker + (1 if info.id < remainder else 0)
        return self.shard_paths[start:end]

    def _generate_chunks(self):
        """Read parquet files, decode rows, and yield fixed-size chunks."""
        shards = self._worker_shards()
        if self.shuffle:
            shards = shards.copy()
            random.shuffle(shards)
        for path in shards:
            table = pq.read_table(path)
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                sample = _decode_row(row)
                yield from _chunk_sample(sample, self.chunk_frames)

    def __iter__(self):
        chunks = self._generate_chunks()
        if self.shuffle:
            chunks = _ShuffleBuffer(chunks, self.shuffle_buffer)
        for chunk in chunks:
            yield _to_tensors(chunk)


def create_parquet_pipeline(
    data_root: str | Path,
    datasets: list[str],
    split: str,
    shuffle: bool = True,
    shuffle_buffer: int = 5000,
    chunk_seconds: float = 10.0,
) -> ParquetFeatureDataset:
    """Build a parquet data pipeline for pre-computed feature datasets.

    Same interface and output format as ``create_webdataset_pipeline``:
    yields ``(mel, onset, vel)`` float32 tensor tuples with fixed
    time-frame dimensions.

    Args:
        data_root: Root directory containing dataset subdirectories.
        datasets: List of dataset names to include.
        split: Data split (e.g. "train").
        shuffle: Whether to shuffle samples (True for training).
        shuffle_buffer: Number of samples in the shuffle buffer.
        chunk_seconds: Duration of each chunk in seconds.

    Returns:
        An IterableDataset yielding ``(mel, onset, vel)`` tensor tuples,
        each with exactly ``int(chunk_seconds * FPS)`` time frames.
    """
    shards = discover_parquet_shards(data_root, datasets, split)
    chunk_frames = int(chunk_seconds * FPS)

    return ParquetFeatureDataset(
        shard_paths=shards,
        chunk_frames=chunk_frames,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
    )
