"""Multi-dataset DataLoader for combining E-GMD + STAR."""

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler


class MultiDatasetLoader:
    """Wraps multiple datasets with weighted sampling."""

    def __init__(
        self,
        datasets: list[Dataset],
        batch_size: int = 32,
        weights: list[float] | None = None,
        num_workers: int = 0,
    ):
        self.concat = ConcatDataset(datasets)
        sizes = [len(d) for d in datasets]
        total = sum(sizes)

        if weights is None:
            weights = [1.0 / len(datasets)] * len(datasets)

        # Build per-sample weights
        sample_weights = []
        for ds_idx, size in enumerate(sizes):
            w = weights[ds_idx] / size
            sample_weights.extend([w] * size)

        sampler = WeightedRandomSampler(sample_weights, num_samples=total)

        self.loader = DataLoader(
            self.concat,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=True,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
