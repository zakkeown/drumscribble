import numpy as np
import torch
import pytest
from datasets import Dataset

from drumscribble.data.parquet import ParquetDataset
from drumscribble.config import NUM_CLASSES, N_MELS, FPS


# --- Legacy format fixture (flat float arrays, pre-chunked) ---

@pytest.fixture
def fake_legacy_dataset():
    """Create an in-memory HF dataset mimicking the legacy parquet schema."""
    n_frames = 625
    rows = {
        "mel": [torch.randn(1, N_MELS, n_frames).flatten().tolist() for _ in range(10)],
        "onset_target": [torch.zeros(NUM_CLASSES, n_frames).flatten().tolist() for _ in range(10)],
        "vel_target": [torch.zeros(NUM_CLASSES, n_frames).flatten().tolist() for _ in range(10)],
        "source": ["egmd"] * 5 + ["star"] * 5,
    }
    return Dataset.from_dict(rows)


# --- Schismaudio format fixture (bytes, full-length recordings) ---

@pytest.fixture
def fake_bytes_dataset():
    """Create an in-memory HF dataset mimicking the schismaudio bytes schema."""
    rows = []
    # 5 egmd rows (30s each = 1875 frames) and 5 star rows (20s = 1250 frames)
    for i in range(10):
        duration = 30.0 if i < 5 else 20.0
        n_frames = int(duration * FPS)
        source = "egmd" if i < 5 else "star"
        rows.append({
            "mel_spectrogram": np.random.randn(N_MELS, n_frames).astype(np.float32).tobytes(),
            "onset_targets": np.zeros((NUM_CLASSES, n_frames), dtype=np.float32).tobytes(),
            "velocity_targets": np.zeros((NUM_CLASSES, n_frames), dtype=np.float32).tobytes(),
            "n_frames": n_frames,
            "n_mels": N_MELS,
            "n_classes": NUM_CLASSES,
            "split": "train" if i < 8 else "validation",
            "augmentation": "",
            "source_audio": f"test_{i}.wav",
        })
    return Dataset.from_list(rows)


# --- Legacy format tests ---

def test_legacy_len(fake_legacy_dataset):
    ds = ParquetDataset(fake_legacy_dataset)
    assert len(ds) == 10


def test_legacy_getitem_shapes(fake_legacy_dataset):
    ds = ParquetDataset(fake_legacy_dataset)
    mel, onset, vel = ds[0]
    assert mel.shape == (1, N_MELS, 625)
    assert onset.shape == (NUM_CLASSES, 625)
    assert vel.shape == (NUM_CLASSES, 625)
    assert mel.dtype == torch.float32


def test_legacy_filter_source(fake_legacy_dataset):
    ds = ParquetDataset(fake_legacy_dataset, source="egmd")
    assert len(ds) == 5


# --- Bytes format tests ---

def test_bytes_len(fake_bytes_dataset):
    """Bytes dataset should be chunked: 30s/10s=3 chunks * 5 + 20s/10s=2 chunks * 5 = 25."""
    ds = ParquetDataset(fake_bytes_dataset, chunk_seconds=10.0)
    assert len(ds) == 25


def test_bytes_getitem_shapes(fake_bytes_dataset):
    ds = ParquetDataset(fake_bytes_dataset, chunk_seconds=10.0)
    chunk_frames = int(10.0 * FPS)
    mel, onset, vel = ds[0]
    assert mel.shape == (1, N_MELS, chunk_frames)
    assert onset.shape == (NUM_CLASSES, chunk_frames)
    assert vel.shape == (NUM_CLASSES, chunk_frames)
    assert mel.dtype == torch.float32


def test_bytes_split_filter(fake_bytes_dataset):
    """Filtering by split='train' should keep 8 of 10 rows."""
    ds = ParquetDataset(fake_bytes_dataset, split="train", chunk_seconds=10.0)
    # 5 egmd (30s, 3 chunks each) + 3 star (20s, 2 chunks each) = 15 + 6 = 21
    assert len(ds) == 21


def test_bytes_empty_after_filter(fake_bytes_dataset):
    with pytest.raises(ValueError, match="empty after filtering"):
        ParquetDataset(fake_bytes_dataset, split="nonexistent")


def test_bytes_dataloader(fake_bytes_dataset):
    from torch.utils.data import DataLoader
    ds = ParquetDataset(fake_bytes_dataset, chunk_seconds=10.0)
    chunk_frames = int(10.0 * FPS)
    loader = DataLoader(ds, batch_size=4)
    mel_batch, onset_batch, vel_batch = next(iter(loader))
    assert mel_batch.shape == (4, 1, N_MELS, chunk_frames)
    assert onset_batch.shape == (4, NUM_CLASSES, chunk_frames)
    assert vel_batch.shape == (4, NUM_CLASSES, chunk_frames)


def test_bytes_training_loop(fake_bytes_dataset):
    """Verify bytes ParquetDataset works end-to-end with the training loop."""
    from torch.utils.data import DataLoader
    from drumscribble.model.drumscribble import DrumscribbleCNN
    from drumscribble.loss import DrumscribbleLoss
    from drumscribble.train import train_one_epoch, create_optimizer

    ds = ParquetDataset(fake_bytes_dataset, split="train", chunk_seconds=10.0)
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
