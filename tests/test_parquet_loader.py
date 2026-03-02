"""Tests for parquet-based feature loader."""
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from drumscribble.config import N_MELS, NUM_CLASSES


def _make_parquet_shard(
    path: Path,
    n_samples: int = 5,
    n_frames: int = 625,
    n_mels: int = N_MELS,
    n_classes: int = NUM_CLASSES,
) -> None:
    """Create a synthetic parquet shard matching the HF dataset schema."""
    rows = []
    for i in range(n_samples):
        mel = np.random.randn(n_mels, n_frames).astype(np.float32)
        onset = np.zeros((n_classes, n_frames), dtype=np.float32)
        vel = np.zeros((n_classes, n_frames), dtype=np.float32)
        rows.append({
            "mel_spectrogram": mel.tobytes(),
            "onset_targets": onset.tobytes(),
            "velocity_targets": vel.tobytes(),
            "n_frames": n_frames,
            "n_mels": n_mels,
            "n_classes": n_classes,
            "sample_rate": 16000,
            "hop_length": 256,
            "fps": 62.5,
            "duration": n_frames / 62.5,
            "split": "train",
            "source_id": f"sample_{i}",
        })

    table = pa.table({
        "mel_spectrogram": pa.array([r["mel_spectrogram"] for r in rows], type=pa.binary()),
        "onset_targets": pa.array([r["onset_targets"] for r in rows], type=pa.binary()),
        "velocity_targets": pa.array([r["velocity_targets"] for r in rows], type=pa.binary()),
        "n_frames": pa.array([r["n_frames"] for r in rows], type=pa.int64()),
        "n_mels": pa.array([r["n_mels"] for r in rows], type=pa.int64()),
        "n_classes": pa.array([r["n_classes"] for r in rows], type=pa.int64()),
        "sample_rate": pa.array([r["sample_rate"] for r in rows], type=pa.int64()),
        "hop_length": pa.array([r["hop_length"] for r in rows], type=pa.int64()),
        "fps": pa.array([r["fps"] for r in rows], type=pa.float64()),
        "duration": pa.array([r["duration"] for r in rows], type=pa.float64()),
        "split": pa.array([r["split"] for r in rows], type=pa.string()),
        "source_id": pa.array([r["source_id"] for r in rows], type=pa.string()),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


@pytest.fixture
def parquet_root(tmp_path):
    """Create a fake parquet data root with e-gmd-aug structure.

    Uses variable-length recordings (1500 frames = 2 chunks + remainder)
    to test chunking. 5 samples * 2 chunks each = 10 chunks.
    """
    feature_dir = tmp_path / "e-gmd-aug" / "features"
    _make_parquet_shard(
        feature_dir / "train-00000.parquet",
        n_samples=5,
        n_frames=1500,
    )
    return tmp_path


def test_discover_parquet_shards(parquet_root):
    from drumscribble.data.parquet_loader import discover_parquet_shards

    shards = discover_parquet_shards(parquet_root, datasets=["e-gmd-aug"], split="train")
    assert len(shards) == 1
    assert "train-00000.parquet" in str(shards[0])


def test_discover_parquet_shards_missing_dataset(parquet_root):
    from drumscribble.data.parquet_loader import discover_parquet_shards

    with pytest.raises(FileNotFoundError):
        discover_parquet_shards(parquet_root, datasets=["nonexistent"], split="train")


def test_decode_row():
    from drumscribble.data.parquet_loader import _decode_row

    n_frames = 100
    mel = np.random.randn(N_MELS, n_frames).astype(np.float32)
    onset = np.random.randn(NUM_CLASSES, n_frames).astype(np.float32)
    vel = np.random.randn(NUM_CLASSES, n_frames).astype(np.float32)

    row = {
        "mel_spectrogram": mel.tobytes(),
        "onset_targets": onset.tobytes(),
        "velocity_targets": vel.tobytes(),
        "n_frames": n_frames,
        "n_mels": N_MELS,
        "n_classes": NUM_CLASSES,
    }
    decoded_mel, decoded_onset, decoded_vel = _decode_row(row)
    assert decoded_mel.shape == (N_MELS, n_frames)
    assert decoded_onset.shape == (NUM_CLASSES, n_frames)
    assert decoded_vel.shape == (NUM_CLASSES, n_frames)
    np.testing.assert_array_equal(decoded_mel, mel)
    np.testing.assert_array_equal(decoded_onset, onset)
    np.testing.assert_array_equal(decoded_vel, vel)


def test_pipeline_shapes(parquet_root):
    from drumscribble.data.parquet_loader import create_parquet_pipeline

    pipeline = create_parquet_pipeline(
        data_root=parquet_root,
        datasets=["e-gmd-aug"],
        split="train",
        shuffle=False,
    )
    sample = next(iter(pipeline))
    mel, onset, vel = sample
    assert mel.shape == (N_MELS, 625), f"Expected (128, 625), got {mel.shape}"
    assert onset.shape == (NUM_CLASSES, 625)
    assert vel.shape == (NUM_CLASSES, 625)
    assert mel.dtype == torch.float32


def test_pipeline_chunks_variable_length(parquet_root):
    """5 samples of 1500 frames -> 2 chunks each (625 remainder dropped) = 10."""
    from drumscribble.data.parquet_loader import create_parquet_pipeline

    pipeline = create_parquet_pipeline(
        data_root=parquet_root,
        datasets=["e-gmd-aug"],
        split="train",
        shuffle=False,
    )
    samples = list(pipeline)
    assert len(samples) == 10


def test_pipeline_pads_short_samples(tmp_path):
    """Samples shorter than chunk_frames should be zero-padded."""
    feature_dir = tmp_path / "e-gmd-aug" / "features"
    _make_parquet_shard(
        feature_dir / "train-00000.parquet",
        n_samples=1,
        n_frames=200,
    )

    from drumscribble.data.parquet_loader import create_parquet_pipeline

    pipeline = create_parquet_pipeline(
        data_root=tmp_path,
        datasets=["e-gmd-aug"],
        split="train",
        shuffle=False,
    )
    samples = list(pipeline)
    assert len(samples) == 1
    mel, onset, vel = samples[0]
    assert mel.shape == (N_MELS, 625)
    # Last 425 frames should be zero (padding)
    assert mel[:, 200:].abs().sum() == 0


def test_dataloader_batching(parquet_root):
    from drumscribble.data.parquet_loader import create_parquet_pipeline

    pipeline = create_parquet_pipeline(
        data_root=parquet_root,
        datasets=["e-gmd-aug"],
        split="train",
        shuffle=False,
    )
    loader = torch.utils.data.DataLoader(pipeline, batch_size=2, num_workers=0)
    mel_batch, onset_batch, vel_batch = next(iter(loader))
    assert mel_batch.shape == (2, N_MELS, 625)
    assert onset_batch.shape == (2, NUM_CLASSES, 625)
    assert vel_batch.shape == (2, NUM_CLASSES, 625)


@pytest.fixture
def multi_parquet_root(tmp_path):
    """Create a parquet data root with two datasets."""
    for ds_name, prefix in [("e-gmd-aug", "egmd"), ("star-drums-aug", "star")]:
        feature_dir = tmp_path / ds_name / "features"
        _make_parquet_shard(
            feature_dir / "train-00000.parquet",
            n_samples=4,
            n_frames=1250,  # 2 chunks each
        )
    return tmp_path


def test_multi_dataset_pipeline(multi_parquet_root):
    """4 samples * 2 chunks * 2 datasets = 16 total chunks."""
    from drumscribble.data.parquet_loader import create_parquet_pipeline

    pipeline = create_parquet_pipeline(
        data_root=multi_parquet_root,
        datasets=["e-gmd-aug", "star-drums-aug"],
        split="train",
        shuffle=False,
    )
    samples = list(pipeline)
    assert len(samples) == 16
    mel, onset, vel = samples[0]
    assert mel.shape == (N_MELS, 625)


def test_training_loop(parquet_root):
    """Full training loop integration with parquet pipeline."""
    from drumscribble.data.augment import SpecAugment
    from drumscribble.data.parquet_loader import create_parquet_pipeline
    from drumscribble.loss import DrumscribbleLoss
    from drumscribble.model.drumscribble import DrumscribbleCNN
    from drumscribble.train import create_optimizer, train_one_epoch

    augment = SpecAugment()

    def collate_fn(batch):
        mels, onsets, vels = zip(*batch)
        mel_batch = torch.stack(mels)
        onset_batch = torch.stack(onsets)
        vel_batch = torch.stack(vels)
        mel_batch = augment(mel_batch)
        return mel_batch, onset_batch, vel_batch

    pipeline = create_parquet_pipeline(
        data_root=parquet_root,
        datasets=["e-gmd-aug"],
        split="train",
        shuffle=False,
    )
    loader = torch.utils.data.DataLoader(
        pipeline, batch_size=2, num_workers=0, collate_fn=collate_fn,
    )

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
