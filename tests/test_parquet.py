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
