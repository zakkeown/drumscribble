import torch
import pytest
from drumscribble.train import train_one_epoch, create_optimizer
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.loss import DrumscribbleLoss


@pytest.fixture
def tiny_model():
    return DrumscribbleCNN(
        backbone_dims=(32, 32, 32, 32),
        backbone_depths=(1, 1, 1, 1),
        num_attn_layers=1,
    )


def test_create_optimizer(tiny_model):
    opt = create_optimizer(tiny_model, lr=1e-3, weight_decay=0.05)
    assert len(opt.param_groups) > 0


def test_train_one_epoch(tiny_model):
    """Verify one epoch runs without error on synthetic data."""
    from torch.utils.data import DataLoader, TensorDataset

    n = 8
    mel = torch.randn(n, 1, 128, 200)
    onset = torch.zeros(n, 26, 200)
    vel = torch.zeros(n, 26, 200)
    ds = TensorDataset(mel, onset, vel)
    loader = DataLoader(ds, batch_size=4)

    opt = create_optimizer(tiny_model, lr=1e-3)
    loss_fn = DrumscribbleLoss()

    avg_loss = train_one_epoch(tiny_model, loader, opt, loss_fn, device="cpu")
    assert avg_loss > 0
    assert not torch.isnan(torch.tensor(avg_loss))
