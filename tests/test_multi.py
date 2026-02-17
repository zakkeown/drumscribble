import torch
from torch.utils.data import TensorDataset
from drumscribble.data.multi import MultiDatasetLoader


def test_multi_dataset_interleaves():
    ds1 = TensorDataset(torch.ones(10, 3), torch.zeros(10))
    ds2 = TensorDataset(torch.ones(5, 3) * 2, torch.ones(5))
    loader = MultiDatasetLoader([ds1, ds2], batch_size=4, weights=[0.5, 0.5])
    batch = next(iter(loader))
    assert batch[0].shape[0] == 4


def test_multi_dataset_epoch_length():
    ds1 = TensorDataset(torch.ones(20, 3), torch.zeros(20))
    ds2 = TensorDataset(torch.ones(10, 3), torch.zeros(10))
    loader = MultiDatasetLoader([ds1, ds2], batch_size=4)
    n_batches = len(list(loader))
    assert n_batches > 0
