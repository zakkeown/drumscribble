"""Dataset that loads pre-computed features from HF datasets Parquet rows."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from drumscribble.config import N_MELS, NUM_CLASSES


def _extract_chunk(
    row: dict,
    n_frames: int,
    base_start: int,
    chunk_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract a randomly-offset chunk from a feature row."""
    max_start = min(base_start + chunk_frames, n_frames - chunk_frames)
    start = (
        torch.randint(base_start, max_start + 1, (1,)).item()
        if max_start > base_start
        else base_start
    )

    mel = np.frombuffer(row["mel_spectrogram"], dtype=np.float32).reshape(N_MELS, n_frames)
    onset = np.frombuffer(row["onset_targets"], dtype=np.float32).reshape(NUM_CLASSES, n_frames)
    vel = np.frombuffer(row["velocity_targets"], dtype=np.float32).reshape(NUM_CLASSES, n_frames)

    mel_chunk = torch.from_numpy(mel[:, start : start + chunk_frames].copy()).unsqueeze(0)
    onset_chunk = torch.from_numpy(onset[:, start : start + chunk_frames].copy())
    vel_chunk = torch.from_numpy(vel[:, start : start + chunk_frames].copy())

    return mel_chunk, onset_chunk, vel_chunk


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
        self.rows: list[dict] = []
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
        return _extract_chunk(row, n_frames, base_start, self.chunk_frames)


class ParquetFeaturesDataset(Dataset):
    """Lazy-loading dataset that reads features from Parquet shards on demand.

    Unlike FeaturesDataset which holds all rows in memory, this only keeps
    a lightweight index of (shard, row, n_frames) metadata. Feature bytes
    are read from Parquet files in __getitem__, with a per-worker shard
    cache to avoid redundant I/O.

    Args:
        shard_paths: List of local Parquet file paths.
        chunk_frames: Number of frames per training chunk (default 625).
    """

    _FEATURE_COLS = ["mel_spectrogram", "onset_targets", "velocity_targets", "n_frames"]

    def __init__(self, shard_paths: list[str], chunk_frames: int = 625) -> None:
        import pyarrow.parquet as pq

        self.shard_paths = list(shard_paths)
        self.chunk_frames = chunk_frames
        # (shard_idx, row_in_shard, n_frames)
        self.row_meta: list[tuple[int, int, int]] = []
        self.chunks: list[tuple[int, int]] = []  # (meta_idx, start_frame)

        for shard_idx, path in enumerate(shard_paths):
            table = pq.read_table(path, columns=["n_frames"])
            for row_in_shard, n_frames in enumerate(
                table.column("n_frames").to_pylist()
            ):
                if n_frames < chunk_frames:
                    continue
                meta_idx = len(self.row_meta)
                self.row_meta.append((shard_idx, row_in_shard, n_frames))
                n_chunks = n_frames // chunk_frames
                for c in range(n_chunks):
                    self.chunks.append((meta_idx, c * chunk_frames))

        self._reset_cache()

    def _reset_cache(self) -> None:
        """Clear the shard cache. Called on init and by worker_init_fn."""
        self._cached_shard_idx: int = -1
        self._cached_table = None

    def __len__(self) -> int:
        return len(self.chunks)

    def _read_row(self, shard_idx: int, row_in_shard: int) -> dict:
        """Read a single row from a shard, caching the current shard."""
        import pyarrow.parquet as pq

        if self._cached_shard_idx != shard_idx:
            self._cached_table = pq.read_table(
                self.shard_paths[shard_idx], columns=self._FEATURE_COLS
            )
            self._cached_shard_idx = shard_idx
        row = self._cached_table.slice(row_in_shard, 1)
        return {col: row.column(col)[0].as_py() for col in self._FEATURE_COLS}

    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        """DataLoader worker_init_fn — resets shard cache after fork."""
        import torch.utils.data as data

        ds = data.get_worker_info().dataset
        if isinstance(ds, ParquetFeaturesDataset):
            ds._reset_cache()

    def shard_for_chunk(self, chunk_idx: int) -> int:
        """Return the shard index for a given chunk index."""
        meta_idx, _ = self.chunks[chunk_idx]
        return self.row_meta[meta_idx][0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        meta_idx, base_start = self.chunks[idx]
        shard_idx, row_in_shard, n_frames = self.row_meta[meta_idx]
        row = self._read_row(shard_idx, row_in_shard)
        return _extract_chunk(row, n_frames, base_start, self.chunk_frames)


class ShardGroupedSampler(Sampler[int]):
    """Sampler that groups chunk indices by shard to minimize Parquet I/O.

    Shuffles the order of shards each epoch, and shuffles chunks within
    each shard, but processes all chunks from one shard before moving to
    the next. This reduces shard reads from O(N) to O(num_shards).
    """

    def __init__(self, dataset: ParquetFeaturesDataset) -> None:
        self.dataset = dataset
        # Group chunk indices by shard
        self._shard_groups: dict[int, list[int]] = defaultdict(list)
        for chunk_idx in range(len(dataset)):
            shard_idx = dataset.shard_for_chunk(chunk_idx)
            self._shard_groups[shard_idx].append(chunk_idx)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        shards = list(self._shard_groups.keys())
        random.shuffle(shards)
        for shard_idx in shards:
            indices = self._shard_groups[shard_idx].copy()
            random.shuffle(indices)
            yield from indices
