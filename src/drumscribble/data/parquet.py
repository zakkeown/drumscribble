"""HF parquet dataset wrapper for pre-computed mel spectrograms."""
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset

from drumscribble.config import FPS, NUM_CLASSES, N_MELS


def _pre_chunk_bytes(hf_dataset, chunk_frames: int) -> Dataset:
    """Materialize bytes-format recordings into pre-chunked rows.

    Converts full-length recordings into fixed-size chunks stored as float16
    bytes, so __getitem__ only loads one chunk's worth of data per access.
    Float16 halves storage (~153 GB vs ~307 GB) with negligible precision loss
    for mel specs and targets. Converted back to float32 at read time.
    """
    all_nframes = hf_dataset["n_frames"]

    def generate_chunks():
        for i, n_frames in enumerate(all_nframes):
            row = hf_dataset[i]
            mel = np.frombuffer(row["mel_spectrogram"], dtype=np.float32).reshape(N_MELS, -1)
            onset = np.frombuffer(row["onset_targets"], dtype=np.float32).reshape(NUM_CLASSES, -1)
            vel = np.frombuffer(row["velocity_targets"], dtype=np.float32).reshape(NUM_CLASSES, -1)

            starts = [0] if n_frames <= chunk_frames else list(
                range(0, n_frames - chunk_frames + 1, chunk_frames)
            )
            for start in starts:
                end = min(start + chunk_frames, n_frames)
                length = end - start

                if length == chunk_frames:
                    mel_chunk = mel[:, start:end]
                    onset_chunk = onset[:, start:end]
                    vel_chunk = vel[:, start:end]
                else:
                    mel_chunk = np.zeros((N_MELS, chunk_frames), dtype=np.float32)
                    onset_chunk = np.zeros((NUM_CLASSES, chunk_frames), dtype=np.float32)
                    vel_chunk = np.zeros((NUM_CLASSES, chunk_frames), dtype=np.float32)
                    mel_chunk[:, :length] = mel[:, start:end]
                    onset_chunk[:, :length] = onset[:, start:end]
                    vel_chunk[:, :length] = vel[:, start:end]

                yield {
                    "mel": mel_chunk.astype(np.float16).tobytes(),
                    "onset": onset_chunk.astype(np.float16).tobytes(),
                    "vel": vel_chunk.astype(np.float16).tobytes(),
                }

    return Dataset.from_generator(generate_chunks)


class ParquetDataset(TorchDataset):
    """Wraps an HF datasets.Dataset of pre-computed mel specs and targets.

    Supports two formats:
    - schismaudio format: bytes columns (mel_spectrogram, onset_targets,
      velocity_targets) with full-length recordings. These are pre-chunked
      on first load into fixed-size rows for fast access.
    - legacy format: flat float arrays (mel, onset_target, vel_target)
      pre-chunked to fixed length.
    """

    def __init__(
        self,
        hf_dataset,
        source: str | None = None,
        split: str | None = None,
        chunk_seconds: float = 10.0,
    ):
        # Filter by source column if present
        if source is not None:
            if "source" in hf_dataset.column_names:
                hf_dataset = hf_dataset.filter(lambda row: row["source"] == source)

        # Filter by split column (schismaudio datasets store split as a column)
        if split is not None and "split" in hf_dataset.column_names:
            hf_dataset = hf_dataset.filter(lambda row: row["split"] == split)

        if len(hf_dataset) == 0:
            raise ValueError(
                f"Dataset is empty after filtering (source={source!r}, split={split!r})."
            )

        # Detect format
        self.is_bytes_format = "mel_spectrogram" in hf_dataset.column_names
        self.chunk_frames = int(chunk_seconds * FPS)

        if self.is_bytes_format:
            # Pre-chunk full recordings into fixed-size rows for fast access
            print(f"Pre-chunking {len(hf_dataset)} recordings into {chunk_seconds}s chunks...")
            self.dataset = _pre_chunk_bytes(hf_dataset, self.chunk_frames)
            print(f"  -> {len(self.dataset)} chunks")
        else:
            # Legacy: pre-chunked, infer frame count from first row
            self.dataset = hf_dataset
            mel_len = len(self.dataset[0]["mel"])
            self.n_frames = mel_len // N_MELS

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_bytes_format:
            return self._get_bytes_item(idx)
        return self._get_legacy_item(idx)

    def _get_bytes_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataset[idx]
        mel = torch.from_numpy(
            np.frombuffer(row["mel"], dtype=np.float16).reshape(1, N_MELS, self.chunk_frames).copy()
        ).float()
        onset = torch.from_numpy(
            np.frombuffer(row["onset"], dtype=np.float16).reshape(NUM_CLASSES, self.chunk_frames).copy()
        ).float()
        vel = torch.from_numpy(
            np.frombuffer(row["vel"], dtype=np.float16).reshape(NUM_CLASSES, self.chunk_frames).copy()
        ).float()
        return mel, onset, vel

    def _get_legacy_item(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataset[idx]
        mel = torch.tensor(row["mel"], dtype=torch.float32).reshape(1, N_MELS, self.n_frames)
        onset = torch.tensor(row["onset_target"], dtype=torch.float32).reshape(NUM_CLASSES, self.n_frames)
        vel = torch.tensor(row["vel_target"], dtype=torch.float32).reshape(NUM_CLASSES, self.n_frames)
        return mel, onset, vel
