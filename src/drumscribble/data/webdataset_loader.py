"""WebDataset loader for pre-computed feature shards."""
from pathlib import Path

import numpy as np
import torch
import webdataset as wds

from drumscribble.config import FPS


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


def _chunk_sample(sample, chunk_frames: int):
    """Split a variable-length sample into fixed-size chunks.

    Yields (mel_chunk, onset_chunk, vel_chunk) tuples where each
    chunk has exactly chunk_frames time frames. Samples shorter than
    chunk_frames are zero-padded. The last chunk of a long sample is
    dropped if shorter than chunk_frames.
    """
    mel, onset, vel = sample
    n_frames = mel.shape[-1]

    if n_frames <= chunk_frames:
        # Pad short samples
        pad_width = chunk_frames - n_frames
        mel = np.pad(mel, ((0, 0), (0, pad_width)))
        onset = np.pad(onset, ((0, 0), (0, pad_width)))
        vel = np.pad(vel, ((0, 0), (0, pad_width)))
        yield (mel, onset, vel)
    else:
        # Split into non-overlapping chunks, drop remainder
        for start in range(0, n_frames - chunk_frames + 1, chunk_frames):
            end = start + chunk_frames
            yield (mel[:, start:end], onset[:, start:end], vel[:, start:end])


def _to_tensors(sample):
    """Convert numpy arrays to float32 tensors."""
    mel, onset, vel = sample
    return (
        torch.from_numpy(mel.copy()).float(),
        torch.from_numpy(onset.copy()).float(),
        torch.from_numpy(vel.copy()).float(),
    )


class _ChunkSamples:
    """Picklable callable that chunks variable-length samples."""

    def __init__(self, chunk_frames: int):
        self.chunk_frames = chunk_frames

    def __call__(self, src):
        for sample in src:
            yield from _chunk_sample(sample, self.chunk_frames)


def create_webdataset_pipeline(
    shard_root: str | Path,
    datasets: list[str],
    split: str,
    shuffle: bool = True,
    shuffle_buffer: int = 5000,
    epoch_size: int | None = None,
    chunk_seconds: float = 10.0,
) -> wds.WebDataset:
    """Build a WebDataset pipeline for feature shards.

    Each shard sample may contain a full-length recording. This pipeline
    chunks each recording into fixed-size segments of chunk_seconds
    before yielding individual training samples.

    Args:
        shard_root: Root directory containing dataset subdirectories.
        datasets: List of dataset names to include.
        split: Data split ("train", "validation", or "test").
        shuffle: Whether to shuffle samples (True for training).
        shuffle_buffer: Number of samples in the shuffle buffer.
        epoch_size: Number of samples per epoch. If None, iterates
            through all samples once.
        chunk_seconds: Duration of each chunk in seconds.

    Returns:
        A WebDataset pipeline (IterableDataset) yielding
        (mel, onset_targets, velocity_targets) tensor tuples,
        each with exactly int(chunk_seconds * FPS) time frames.
    """
    shards = discover_shards(shard_root, datasets, split)
    chunk_frames = int(chunk_seconds * FPS)

    pipeline = wds.WebDataset(
        shards, shardshuffle=len(shards) if shuffle else False
    )

    pipeline = (
        pipeline
        .decode()
        .to_tuple(
            "mel_spectrogram.npy",
            "onset_targets.npy",
            "velocity_targets.npy",
        )
        .compose(_ChunkSamples(chunk_frames))
    )

    if shuffle:
        pipeline = pipeline.shuffle(shuffle_buffer)

    pipeline = pipeline.map(_to_tensors)

    if epoch_size is not None:
        pipeline = pipeline.with_epoch(epoch_size)

    return pipeline
