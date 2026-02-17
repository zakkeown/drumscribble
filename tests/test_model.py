import torch
from drumscribble.model.drumscribble import DrumscribbleCNN, UNetDecoderBlock
from drumscribble.config import NUM_CLASSES


def test_decoder_block_shape():
    block = UNetDecoderBlock(in_ch=384, skip_ch=256, out_ch=256)
    x = torch.randn(2, 384, 1, 78)
    skip = torch.randn(2, 256, 1, 156)
    out = block(x, skip)
    assert out.shape == (2, 256, 1, 156)


def test_decoder_block_odd_sizes():
    """Decoder must handle non-power-of-2 sizes via interpolation to skip size."""
    block = UNetDecoderBlock(in_ch=128, skip_ch=64, out_ch=64)
    x = torch.randn(1, 128, 1, 156)
    skip = torch.randn(1, 64, 1, 313)
    out = block(x, skip)
    assert out.shape == (1, 64, 1, 313)


def test_model_output_shape_10s():
    model = DrumscribbleCNN()
    x = torch.randn(2, 1, 128, 625)
    onset, velocity, offset = model(x)
    assert onset.shape == (2, NUM_CLASSES, 625)
    assert velocity.shape == (2, NUM_CLASSES, 625)
    assert offset.shape == (2, NUM_CLASSES, 625)


def test_model_output_shape_30s():
    model = DrumscribbleCNN()
    x = torch.randn(1, 1, 128, 1875)
    onset, velocity, offset = model(x)
    assert onset.shape == (1, NUM_CLASSES, 1875)


def test_model_outputs_are_probabilities():
    model = DrumscribbleCNN()
    x = torch.randn(1, 1, 128, 200)
    onset, velocity, offset = model(x)
    assert onset.min() >= 0 and onset.max() <= 1
    assert velocity.min() >= 0 and velocity.max() <= 1
    assert offset.min() >= 0 and offset.max() <= 1


def test_model_with_mert():
    model = DrumscribbleCNN(mert_dim=768)
    x = torch.randn(1, 1, 128, 625)
    mert = torch.randn(1, 768, 1, 500)  # MERT at ~50Hz, FiLM interpolates to T/8
    onset, _, _ = model(x, mert_features=mert)
    assert onset.shape == (1, NUM_CLASSES, 625)


def test_model_without_mert():
    model = DrumscribbleCNN(mert_dim=768)
    x = torch.randn(1, 1, 128, 625)
    onset, _, _ = model(x, mert_features=None)
    assert onset.shape == (1, NUM_CLASSES, 625)


def test_model_param_count():
    model = DrumscribbleCNN()
    total = sum(p.numel() for p in model.parameters())
    assert 10_000_000 < total < 15_000_000  # ~12.1M target


def test_model_returns_tuple():
    """Must return plain tuple for torch.jit.trace compatibility."""
    model = DrumscribbleCNN()
    x = torch.randn(1, 1, 128, 200)
    result = model(x)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_model_frozen_bn():
    model = DrumscribbleCNN()
    model.freeze_bn()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            assert not m.training


def test_model_skip_connections_used():
    """Verify the model uses skip connections (output should differ from no-skip baseline)."""
    model = DrumscribbleCNN()
    model.eval()
    x = torch.randn(1, 1, 128, 200)
    with torch.no_grad():
        onset, _, _ = model(x)
    assert onset.shape[-1] == 200
