"""Tests for WebDataset-based shard loader."""
import io
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from drumscribble.config import N_MELS, NUM_CLASSES


def _make_shard(path: Path, keys: list[str], n_frames: int = 625) -> None:
    """Create a synthetic feature shard tar at *path*."""
    with tarfile.open(path, "w") as tar:
        for key in keys:
            mel = np.random.randn(N_MELS, n_frames).astype(np.float32)
            onset = np.zeros((NUM_CLASSES, n_frames), dtype=np.float32)
            vel = np.zeros((NUM_CLASSES, n_frames), dtype=np.float32)
            params = {"sr": 16000, "hop_length": 256, "n_mels": 128,
                       "n_classes": NUM_CLASSES, "fps": 62.5}

            for suffix, data in [
                ("mel_spectrogram.npy", mel),
                ("onset_targets.npy", onset),
                ("velocity_targets.npy", vel),
            ]:
                buf = io.BytesIO()
                np.save(buf, data)
                buf.seek(0)
                info = tarfile.TarInfo(name=f"{key}.{suffix}")
                info.size = buf.getbuffer().nbytes
                tar.addfile(info, buf)

            # params.json
            params_bytes = json.dumps(params).encode()
            info = tarfile.TarInfo(name=f"{key}.params.json")
            info.size = len(params_bytes)
            tar.addfile(info, io.BytesIO(params_bytes))


@pytest.fixture
def shard_root(tmp_path):
    """Create a fake shard_root with egmd_upload structure.

    Uses variable-length recordings (1500 frames = 2 chunks + remainder)
    to test chunking. 5 samples * 2 chunks each = 10 chunks for training.
    3 validation samples * 2 chunks each = 6 chunks.
    """
    train_dir = tmp_path / "egmd_upload" / "data" / "features" / "train"
    train_dir.mkdir(parents=True)
    _make_shard(train_dir / "feature-shard-00000.tar",
                keys=[f"sample_{i}" for i in range(5)],
                n_frames=1500)  # 2 full chunks of 625 + 250 remainder (dropped)

    val_dir = tmp_path / "egmd_upload" / "data" / "features" / "validation"
    val_dir.mkdir(parents=True)
    _make_shard(val_dir / "feature-shard-00000.tar",
                keys=[f"val_{i}" for i in range(3)],
                n_frames=1500)
    return tmp_path


def test_discover_shards(shard_root):
    from drumscribble.data.webdataset_loader import discover_shards

    shards = discover_shards(shard_root, datasets=["egmd_upload"], split="train")
    assert len(shards) == 1
    assert "feature-shard-00000.tar" in str(shards[0])


def test_discover_shards_missing_dataset(shard_root):
    from drumscribble.data.webdataset_loader import discover_shards

    with pytest.raises(FileNotFoundError):
        discover_shards(shard_root, datasets=["nonexistent"], split="train")


def test_pipeline_shapes(shard_root):
    from drumscribble.data.webdataset_loader import create_webdataset_pipeline

    pipeline = create_webdataset_pipeline(
        shard_root=shard_root,
        datasets=["egmd_upload"],
        split="train",
        shuffle=False,
    )
    sample = next(iter(pipeline))
    mel, onset, vel = sample
    assert mel.shape == (N_MELS, 625), f"Expected (128, 625), got {mel.shape}"
    assert onset.shape == (NUM_CLASSES, 625)
    assert vel.shape == (NUM_CLASSES, 625)
    assert mel.dtype == torch.float32


def test_pipeline_chunks_variable_length(shard_root):
    """5 samples of 1500 frames -> 2 chunks each (625 remainder dropped) = 10."""
    from drumscribble.data.webdataset_loader import create_webdataset_pipeline

    pipeline = create_webdataset_pipeline(
        shard_root=shard_root,
        datasets=["egmd_upload"],
        split="train",
        shuffle=False,
    )
    samples = list(pipeline)
    assert len(samples) == 10


def test_pipeline_pads_short_samples(tmp_path):
    """Samples shorter than chunk_frames should be zero-padded."""
    train_dir = tmp_path / "egmd_upload" / "data" / "features" / "train"
    train_dir.mkdir(parents=True)
    _make_shard(train_dir / "feature-shard-00000.tar",
                keys=["short_0"], n_frames=200)

    from drumscribble.data.webdataset_loader import create_webdataset_pipeline

    pipeline = create_webdataset_pipeline(
        shard_root=tmp_path, datasets=["egmd_upload"],
        split="train", shuffle=False,
    )
    samples = list(pipeline)
    assert len(samples) == 1
    mel, onset, vel = samples[0]
    assert mel.shape == (N_MELS, 625)
    # Last 425 frames should be zero (padding)
    assert mel[:, 200:].abs().sum() == 0


def test_pipeline_validation_split(shard_root):
    """3 validation samples of 1500 frames -> 2 chunks each = 6."""
    from drumscribble.data.webdataset_loader import create_webdataset_pipeline

    pipeline = create_webdataset_pipeline(
        shard_root=shard_root,
        datasets=["egmd_upload"],
        split="validation",
        shuffle=False,
    )
    samples = list(pipeline)
    assert len(samples) == 6


def test_dataloader_batching(shard_root):
    from drumscribble.data.webdataset_loader import create_webdataset_pipeline

    pipeline = create_webdataset_pipeline(
        shard_root=shard_root,
        datasets=["egmd_upload"],
        split="train",
        shuffle=False,
        epoch_size=4,
    )
    loader = torch.utils.data.DataLoader(pipeline, batch_size=2, num_workers=0)
    mel_batch, onset_batch, vel_batch = next(iter(loader))
    assert mel_batch.shape == (2, N_MELS, 625)
    assert onset_batch.shape == (2, NUM_CLASSES, 625)
    assert vel_batch.shape == (2, NUM_CLASSES, 625)


def test_training_loop(shard_root):
    """Full training loop with WebDataset pipeline."""
    from drumscribble.data.webdataset_loader import create_webdataset_pipeline
    from drumscribble.data.augment import SpecAugment
    from drumscribble.model.drumscribble import DrumscribbleCNN
    from drumscribble.loss import DrumscribbleLoss
    from drumscribble.train import train_one_epoch, create_optimizer

    augment = SpecAugment()

    def collate_fn(batch):
        mels, onsets, vels = zip(*batch)
        mel_batch = torch.stack(mels)
        onset_batch = torch.stack(onsets)
        vel_batch = torch.stack(vels)
        mel_batch = augment(mel_batch)
        return mel_batch, onset_batch, vel_batch

    pipeline = create_webdataset_pipeline(
        shard_root=shard_root,
        datasets=["egmd_upload"],
        split="train",
        shuffle=False,
        epoch_size=4,  # 10 chunks available, use 4 for speed
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


@pytest.fixture
def multi_shard_root(tmp_path):
    """Create a shard_root with two datasets (variable length)."""
    for ds_name, prefix in [("egmd_upload", "egmd"), ("star-drums", "star")]:
        train_dir = tmp_path / ds_name / "data" / "features" / "train"
        train_dir.mkdir(parents=True)
        _make_shard(train_dir / "feature-shard-00000.tar",
                    keys=[f"{prefix}_{i}" for i in range(4)],
                    n_frames=1250)  # 2 chunks each
    return tmp_path


def test_multi_dataset_pipeline(multi_shard_root):
    """4 samples * 2 chunks * 2 datasets = 16 total chunks."""
    from drumscribble.data.webdataset_loader import create_webdataset_pipeline

    pipeline = create_webdataset_pipeline(
        shard_root=multi_shard_root,
        datasets=["egmd_upload", "star-drums"],
        split="train",
        shuffle=False,
        epoch_size=16,
    )
    samples = list(pipeline)
    assert len(samples) == 16
    mel, onset, vel = samples[0]
    assert mel.shape == (N_MELS, 625)
