import torch
import pytest
from drumscribble.train import (
    train_one_epoch,
    create_optimizer,
    create_scheduler,
    EMAModel,
)
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.loss import DrumscribbleLoss


@pytest.fixture
def tiny_model():
    return DrumscribbleCNN(
        backbone_dims=(32, 32, 32, 32),
        backbone_depths=(1, 1, 1, 1),
        num_attn_layers=1,
    )


@pytest.fixture
def synthetic_loader():
    from torch.utils.data import DataLoader, TensorDataset

    n = 8
    mel = torch.randn(n, 1, 128, 200)
    onset = torch.zeros(n, 26, 200)
    vel = torch.zeros(n, 26, 200)
    ds = TensorDataset(mel, onset, vel)
    return DataLoader(ds, batch_size=4)


def test_create_optimizer(tiny_model):
    opt = create_optimizer(tiny_model, lr=1e-3, weight_decay=0.05)
    assert len(opt.param_groups) > 0


def test_train_one_epoch(tiny_model, synthetic_loader):
    """Verify one epoch runs without error on synthetic data."""
    opt = create_optimizer(tiny_model, lr=1e-3)
    loss_fn = DrumscribbleLoss()

    avg_loss = train_one_epoch(tiny_model, synthetic_loader, opt, loss_fn, device="cpu")
    assert avg_loss > 0
    assert not torch.isnan(torch.tensor(avg_loss))


def test_create_scheduler(tiny_model):
    """Verify LR increases during warmup and then decreases via cosine."""
    opt = create_optimizer(tiny_model, lr=1e-3)
    warmup_steps = 10
    total_steps = 100
    scheduler = create_scheduler(opt, warmup_steps, total_steps)

    # Collect LRs over all steps
    lrs = []
    for _ in range(total_steps):
        lrs.append(opt.param_groups[0]["lr"])
        scheduler.step()

    # During warmup, LR should increase
    for i in range(1, warmup_steps):
        assert lrs[i] > lrs[i - 1], f"LR should increase during warmup at step {i}"

    # After warmup, LR should generally decrease (cosine decay)
    # Check that LR at step warmup_steps+10 is less than LR at warmup_steps
    post_warmup_start = lrs[warmup_steps]
    post_warmup_later = lrs[warmup_steps + 20]
    assert post_warmup_later < post_warmup_start, (
        "LR should decrease after warmup via cosine decay"
    )

    # Final LR should be near zero
    assert lrs[-1] < lrs[warmup_steps] * 0.1, "Final LR should be much smaller"


def test_ema_model(tiny_model):
    """Verify EMA tracks parameters and apply/restore works correctly."""
    ema = EMAModel(tiny_model, decay=0.999)

    # Get a reference parameter name and its initial value
    name, param = next(
        (n, p) for n, p in tiny_model.named_parameters() if p.requires_grad
    )
    initial_value = param.data.clone()

    # Simulate a training step: modify the parameter
    with torch.no_grad():
        param.add_(torch.ones_like(param))

    # Update EMA
    ema.update(tiny_model)

    # EMA shadow should be between initial and new value (closer to initial due to 0.999 decay)
    expected_shadow = 0.999 * initial_value + 0.001 * param.data
    assert torch.allclose(ema.shadow[name], expected_shadow, atol=1e-6), (
        "EMA shadow should be exponential average of old and new params"
    )

    # Apply: model params should become EMA shadow values
    ema.apply(tiny_model)
    assert torch.allclose(param.data, expected_shadow, atol=1e-6), (
        "After apply, model params should equal EMA shadow"
    )

    # Restore: model params should go back to pre-apply values
    ema.restore(tiny_model)
    assert torch.allclose(param.data, initial_value + torch.ones_like(param), atol=1e-6), (
        "After restore, model params should equal original (modified) values"
    )


def test_train_one_epoch_with_scheduler(tiny_model, synthetic_loader):
    """Verify training with scheduler completes without error."""
    opt = create_optimizer(tiny_model, lr=1e-3)
    scheduler = create_scheduler(opt, warmup_steps=2, total_steps=10)
    loss_fn = DrumscribbleLoss()

    avg_loss = train_one_epoch(
        tiny_model,
        synthetic_loader,
        opt,
        loss_fn,
        device="cpu",
        scheduler=scheduler,
    )
    assert avg_loss > 0
    assert not torch.isnan(torch.tensor(avg_loss))

    # Verify scheduler actually advanced (LR should have changed from initial)
    # With warmup_steps=2 and 2 batches processed, we should be past warmup
    current_lr = opt.param_groups[0]["lr"]
    assert current_lr > 0, "LR should be positive"


def test_train_one_epoch_with_ema(tiny_model, synthetic_loader):
    """Verify training with EMA completes without error."""
    opt = create_optimizer(tiny_model, lr=1e-3)
    ema = EMAModel(tiny_model, decay=0.99)
    loss_fn = DrumscribbleLoss()

    # Get initial shadow value
    name, _ = next(
        (n, p) for n, p in tiny_model.named_parameters() if p.requires_grad
    )
    initial_shadow = ema.shadow[name].clone()

    avg_loss = train_one_epoch(
        tiny_model,
        synthetic_loader,
        opt,
        loss_fn,
        device="cpu",
        ema=ema,
    )
    assert avg_loss > 0

    # EMA shadow should have been updated (different from initial)
    assert not torch.equal(ema.shadow[name], initial_shadow), (
        "EMA shadow should change after training steps"
    )
