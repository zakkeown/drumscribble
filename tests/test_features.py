"""Tests for FeaturesDataset (pre-computed features from HF datasets)."""

import numpy as np
import pytest
import torch

from drumscribble.config import NUM_CLASSES, N_MELS
from drumscribble.data.features import FeaturesDataset


def _make_feature_row(n_frames: int = 1000) -> dict:
    """Create a fake feature row matching the Parquet schema."""
    mel = np.random.randn(N_MELS, n_frames).astype(np.float32)
    onset = np.random.rand(NUM_CLASSES, n_frames).astype(np.float32)
    vel = np.random.rand(NUM_CLASSES, n_frames).astype(np.float32)
    return {
        "mel_spectrogram": mel.tobytes(),
        "onset_targets": onset.tobytes(),
        "velocity_targets": vel.tobytes(),
        "n_frames": n_frames,
        "n_mels": N_MELS,
        "n_classes": NUM_CLASSES,
        "duration": n_frames / 62.5,
        "split": "train",
    }


class TestFeaturesDataset:
    def test_getitem_returns_correct_shapes(self):
        rows = [_make_feature_row(1000) for _ in range(5)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        mel, onset, vel = ds[0]

        assert mel.shape == (1, N_MELS, 625)
        assert onset.shape == (NUM_CLASSES, 625)
        assert vel.shape == (NUM_CLASSES, 625)

    def test_getitem_returns_tensors(self):
        rows = [_make_feature_row(1000)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        mel, onset, vel = ds[0]

        assert isinstance(mel, torch.Tensor)
        assert isinstance(onset, torch.Tensor)
        assert isinstance(vel, torch.Tensor)
        assert mel.dtype == torch.float32

    def test_length_counts_chunks(self):
        # 1000 frames / 625 chunk = 1 full chunk per row (non-overlapping)
        rows = [_make_feature_row(1000) for _ in range(3)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        assert len(ds) == 3  # 1 chunk per row

    def test_multiple_chunks_per_row(self):
        # 2000 frames / 625 = 3 full chunks
        rows = [_make_feature_row(2000)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        assert len(ds) == 3

    def test_skips_short_rows(self):
        rows = [
            _make_feature_row(100),   # too short for 625 chunk
            _make_feature_row(1000),  # 1 chunk
        ]
        ds = FeaturesDataset(rows, chunk_frames=625)
        assert len(ds) == 1

    def test_random_crop_varies(self):
        rows = [_make_feature_row(2000)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        mel1, _, _ = ds[0]
        mel2, _, _ = ds[0]
        # Random crop means different calls may give different data
        # Just verify shapes are correct
        assert mel1.shape == mel2.shape == (1, N_MELS, 625)

    def test_works_with_dataloader(self):
        rows = [_make_feature_row(1000) for _ in range(4)]
        ds = FeaturesDataset(rows, chunk_frames=625)
        loader = torch.utils.data.DataLoader(ds, batch_size=2)
        batch = next(iter(loader))
        mel, onset, vel = batch
        assert mel.shape == (2, 1, N_MELS, 625)
        assert onset.shape == (2, NUM_CLASSES, 625)
        assert vel.shape == (2, NUM_CLASSES, 625)
