import torch
from drumscribble.loss import DrumscribbleLoss


def test_loss_basic():
    loss_fn = DrumscribbleLoss()
    # Pass raw logits (no sigmoid) — loss applies sigmoid internally
    onset_logits = torch.randn(2, 26, 100)
    vel_logits = torch.randn(2, 26, 100)
    offset_logits = torch.randn(2, 26, 100)
    onset_target = torch.zeros(2, 26, 100)
    vel_target = torch.zeros(2, 26, 100)

    total, components = loss_fn(
        onset_logits, vel_logits, offset_logits, onset_target, vel_target
    )
    assert total.dim() == 0  # scalar
    assert total.item() > 0
    assert "onset" in components
    assert "velocity" in components
    assert "offset" in components


def test_velocity_loss_masked():
    """Velocity loss only computed where onset > 0."""
    loss_fn = DrumscribbleLoss()
    onset_target = torch.zeros(1, 26, 100)
    vel_target = torch.zeros(1, 26, 100)

    # Set one onset
    onset_target[0, 0, 50] = 1.0
    vel_target[0, 0, 50] = 0.8

    # Large positive logit = high probability onset
    onset_logits = torch.full((1, 26, 100), -10.0)
    onset_logits[0, 0, 50] = 10.0
    vel_logits = torch.full((1, 26, 100), -10.0)  # ~0 after sigmoid
    offset_logits = torch.zeros(1, 26, 100)

    _, components = loss_fn(onset_logits, vel_logits, offset_logits, onset_target, vel_target)
    # Velocity loss should be > 0 (predicted ~0, target 0.8)
    assert components["velocity"].item() > 0


def test_loss_zero_when_perfect():
    loss_fn = DrumscribbleLoss()
    target = torch.zeros(1, 26, 50)
    # Large negative logits -> sigmoid ~0 -> matches zero target
    logits = torch.full((1, 26, 50), -10.0)
    total, _ = loss_fn(logits, logits, logits, target, target)
    assert total.item() < 0.01
