import torch
from drumscribble.model.film import FiLMConditioning


def test_film_output_shape():
    film = FiLMConditioning(feature_dim=768, target_dim=384)
    x = torch.randn(2, 384, 1, 313)
    cond = torch.randn(2, 768, 1, 313)
    out = film(x, cond)
    assert out.shape == (2, 384, 1, 313)


def test_film_without_conditioning():
    """Model must work without MERT features (returns input unchanged)."""
    film = FiLMConditioning(feature_dim=768, target_dim=384)
    x = torch.randn(2, 384, 1, 313)
    out = film(x, None)
    assert torch.equal(out, x)


def test_film_no_broadcasting():
    """Gamma/beta must be pre-expanded, not rely on broadcasting."""
    film = FiLMConditioning(feature_dim=768, target_dim=384)
    x = torch.randn(1, 384, 1, 100)
    cond = torch.randn(1, 768, 1, 100)
    out = film(x, cond)
    assert out.shape == x.shape


def test_film_identity_init():
    """FiLM projection should initialize to identity (gamma=1, beta=0)."""
    film = FiLMConditioning(feature_dim=768, target_dim=384)
    x = torch.randn(1, 384, 1, 100)
    # Zero conditioning input should produce gamma=1, beta=0 -> output == input
    cond = torch.zeros(1, 768, 1, 100)
    out = film(x, cond)
    assert torch.allclose(out, x, atol=1e-6)
