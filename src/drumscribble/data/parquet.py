"""HF parquet dataset wrapper for pre-computed mel spectrograms."""
import torch
from torch.utils.data import Dataset as TorchDataset

from drumscribble.config import NUM_CLASSES, N_MELS


class ParquetDataset(TorchDataset):
    """Wraps an HF datasets.Dataset of pre-computed mel specs and targets.

    Each row contains flattened float arrays that get reshaped to tensors:
    - mel: (1, N_MELS, T)
    - onset_target: (NUM_CLASSES, T)
    - vel_target: (NUM_CLASSES, T)
    """

    def __init__(self, hf_dataset, source: str | None = None):
        if source is not None:
            hf_dataset = hf_dataset.filter(lambda row: row["source"] == source)
        self.dataset = hf_dataset
        if len(self.dataset) == 0:
            raise ValueError(
                f"Dataset is empty after filtering (source={source!r}). "
                "Check that the source value matches the data."
            )
        # Infer frame count from first row
        mel_len = len(self.dataset[0]["mel"])
        self.n_frames = mel_len // N_MELS

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.dataset[idx]
        mel = torch.tensor(row["mel"], dtype=torch.float32).reshape(1, N_MELS, self.n_frames)
        onset = torch.tensor(row["onset_target"], dtype=torch.float32).reshape(NUM_CLASSES, self.n_frames)
        vel = torch.tensor(row["vel_target"], dtype=torch.float32).reshape(NUM_CLASSES, self.n_frames)
        return mel, onset, vel
