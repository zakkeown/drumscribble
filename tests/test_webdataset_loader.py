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
    """Create a fake shard_root with egmd_upload structure."""
    train_dir = tmp_path / "egmd_upload" / "data" / "features" / "train"
    train_dir.mkdir(parents=True)
    _make_shard(train_dir / "feature-shard-00000.tar",
                keys=[f"sample_{i}" for i in range(5)])

    val_dir = tmp_path / "egmd_upload" / "data" / "features" / "validation"
    val_dir.mkdir(parents=True)
    _make_shard(val_dir / "feature-shard-00000.tar",
                keys=[f"val_{i}" for i in range(3)])
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
        epoch_size=5,
    )
    sample = next(iter(pipeline))
    mel, onset, vel = sample
    assert mel.shape == (N_MELS, 625), f"Expected (128, 625), got {mel.shape}"
    assert onset.shape == (NUM_CLASSES, 625)
    assert vel.shape == (NUM_CLASSES, 625)
    assert mel.dtype == torch.float32


def test_pipeline_iterates_all_samples(shard_root):
    from drumscribble.data.webdataset_loader import create_webdataset_pipeline

    pipeline = create_webdataset_pipeline(
        shard_root=shard_root,
        datasets=["egmd_upload"],
        split="train",
        shuffle=False,
        epoch_size=5,
    )
    samples = list(pipeline)
    assert len(samples) == 5


def test_pipeline_validation_split(shard_root):
    from drumscribble.data.webdataset_loader import create_webdataset_pipeline

    pipeline = create_webdataset_pipeline(
        shard_root=shard_root,
        datasets=["egmd_upload"],
        split="validation",
        shuffle=False,
        epoch_size=3,
    )
    samples = list(pipeline)
    assert len(samples) == 3


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
