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

        # Random offset within this chunk's window, clamped to valid range
        max_start = min(base_start + self.chunk_frames, n_frames - self.chunk_frames)
        start = (
            torch.randint(base_start, max_start + 1, (1,)).item()
            if max_start > base_start
            else base_start
        )

        mel = np.frombuffer(row["mel_spectrogram"], dtype=np.float32).reshape(N_MELS, n_frames)
        onset = np.frombuffer(row["onset_targets"], dtype=np.float32).reshape(NUM_CLASSES, n_frames)
        vel = np.frombuffer(row["velocity_targets"], dtype=np.float32).reshape(NUM_CLASSES, n_frames)

        mel_chunk = torch.from_numpy(mel[:, start : start + self.chunk_frames].copy()).unsqueeze(0)
        onset_chunk = torch.from_numpy(onset[:, start : start + self.chunk_frames].copy())
        vel_chunk = torch.from_numpy(vel[:, start : start + self.chunk_frames].copy())

        return mel_chunk, onset_chunk, vel_chunk
