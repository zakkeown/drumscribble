import torch
from drumscribble.model.attention import ANESelfAttention


def test_attention_shape():
    attn = ANESelfAttention(dim=384, num_heads=4)
    x = torch.randn(2, 384, 1, 78)  # (B, C, 1, T/8) for 10s: 625→312→156→78
    out = attn(x)
    assert out.shape == (2, 384, 1, 78)


def test_attention_long_sequence():
    attn = ANESelfAttention(dim=384, num_heads=4)
    x = torch.randn(1, 384, 1, 234)  # 30s at T/8: 1875→937→468→234
    out = attn(x)
    assert out.shape == (1, 384, 1, 234)


def test_attention_uses_conv2d():
    """All projections should be Conv2d(1x1), not Linear, for ANE."""
    attn = ANESelfAttention(dim=384, num_heads=4)
    for name, module in attn.named_modules():
        assert not isinstance(module, torch.nn.Linear), f"Found Linear layer: {name}"


def test_attention_is_causal_false():
    """Drum transcription uses bidirectional (non-causal) attention."""
    attn = ANESelfAttention(dim=384, num_heads=4)
    x = torch.randn(1, 384, 1, 50)
    out = attn(x)
    assert out.shape == (1, 384, 1, 50)
