import torch
import pytest


def _has_transformers() -> bool:
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _has_transformers(), reason="transformers not installed")
def test_mert_extractor_shape():
    from drumscribble.mert import MERTExtractor

    extractor = MERTExtractor(layer_indices=[5, 6])
    waveform = torch.randn(1, 16000 * 5)  # 5s at 16kHz
    features = extractor(waveform)
    # MERT-95M outputs 768-dim features
    assert features.shape[1] == 768
    assert features.dim() == 4  # (B, C, 1, T)


@pytest.mark.slow
@pytest.mark.skipif(not _has_transformers(), reason="transformers not installed")
def test_mert_extractor_batch():
    """MERT extractor handles batched input."""
    from drumscribble.mert import MERTExtractor

    extractor = MERTExtractor(layer_indices=[5, 6])
    waveform = torch.randn(2, 16000 * 3)  # 2 clips, 3s each
    features = extractor(waveform)
    assert features.shape[0] == 2
    assert features.shape[1] == 768
    assert features.dim() == 4


@pytest.mark.slow
@pytest.mark.skipif(not _has_transformers(), reason="transformers not installed")
def test_mert_extractor_no_grad():
    """MERT parameters should be frozen (no gradients)."""
    from drumscribble.mert import MERTExtractor

    extractor = MERTExtractor(layer_indices=[5, 6])
    for param in extractor.model.parameters():
        assert not param.requires_grad


def test_mert_module_importable():
    """The mert module should be importable even without transformers."""
    from drumscribble import mert  # noqa: F401

    assert hasattr(mert, "MERTExtractor")
