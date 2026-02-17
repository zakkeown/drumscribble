import torch
from drumscribble.loss import DrumscribbleLoss


def test_loss_basic():
    loss_fn = DrumscribbleLoss()
    onset_pred = torch.sigmoid(torch.randn(2, 26, 100))
    vel_pred = torch.sigmoid(torch.randn(2, 26, 100))
    offset_pred = torch.sigmoid(torch.randn(2, 26, 100))
    onset_target = torch.zeros(2, 26, 100)
    vel_target = torch.zeros(2, 26, 100)

    total, components = loss_fn(
        onset_pred, vel_pred, offset_pred, onset_target, vel_target
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

    pred_onset = onset_target.clone()
    pred_vel = torch.zeros_like(vel_target)  # wrong velocity
    pred_offset = torch.zeros(1, 26, 100)

    _, components = loss_fn(pred_onset, pred_vel, pred_offset, onset_target, vel_target)
    # Velocity loss should be > 0 (predicted 0, target 0.8)
    assert components["velocity"].item() > 0


def test_loss_zero_when_perfect():
    loss_fn = DrumscribbleLoss()
    target = torch.zeros(1, 26, 50)
    pred = torch.zeros(1, 26, 50)
    total, _ = loss_fn(pred, pred, pred, target, target)
    # BCE(0,0) should be very small
    assert total.item() < 0.01
