import pytest
import torch


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def batch_mel():
    """Fake mel spectrogram: (B=2, 1, 128 mel bins, T=625 frames = 10s at 62.5fps)."""
    return torch.randn(2, 1, 128, 625)
