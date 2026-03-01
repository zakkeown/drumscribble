"""WebDataset loader for pre-computed feature shards."""
from pathlib import Path

import torch
import webdataset as wds


def discover_shards(
    shard_root: str | Path,
    datasets: list[str],
    split: str,
) -> list[str]:
    """Find feature shard tar files under shard_root.

    Expects directory structure:
        {shard_root}/{dataset}/data/features/{split}/feature-shard-*.tar

    Args:
        shard_root: Root directory containing dataset subdirectories.
        datasets: List of dataset directory names (e.g. ["egmd_upload"]).
        split: Data split ("train", "validation", or "test").

    Returns:
        Sorted list of absolute shard tar paths as strings.

    Raises:
        FileNotFoundError: If a dataset's feature directory doesn't exist.
    """
    root = Path(shard_root).expanduser()
    all_shards = []
    for name in datasets:
        feature_dir = root / name / "data" / "features" / split
        if not feature_dir.exists():
            raise FileNotFoundError(
                f"Shard directory not found: {feature_dir}"
            )
        shards = sorted(feature_dir.glob("feature-shard-*.tar"))
        all_shards.extend(str(s) for s in shards)
    return all_shards


def _to_tensors(sample):
    """Convert numpy arrays to float32 tensors."""
    mel, onset, vel = sample
    return (
        torch.from_numpy(mel).float(),
        torch.from_numpy(onset).float(),
        torch.from_numpy(vel).float(),
    )


def create_webdataset_pipeline(
    shard_root: str | Path,
    datasets: list[str],
    split: str,
    shuffle: bool = True,
    shuffle_buffer: int = 5000,
    epoch_size: int | None = None,
) -> wds.WebDataset:
    """Build a WebDataset pipeline for feature shards.

    Args:
        shard_root: Root directory containing dataset subdirectories.
        datasets: List of dataset names to include.
        split: Data split ("train", "validation", or "test").
        shuffle: Whether to shuffle samples (True for training).
        shuffle_buffer: Number of samples in the shuffle buffer.
        epoch_size: Number of samples per epoch. If None, iterates
            through all samples once.

    Returns:
        A WebDataset pipeline (IterableDataset) yielding
        (mel, onset_targets, velocity_targets) tensor tuples.
    """
    shards = discover_shards(shard_root, datasets, split)

    pipeline = wds.WebDataset(shards)

    if shuffle:
        pipeline = pipeline.shuffle(shuffle_buffer)

    pipeline = (
        pipeline
        .decode()
        .to_tuple(
            "mel_spectrogram.npy",
            "onset_targets.npy",
            "velocity_targets.npy",
        )
        .map(_to_tensors)
    )

    if epoch_size is not None:
        pipeline = pipeline.with_epoch(epoch_size)

    return pipeline
