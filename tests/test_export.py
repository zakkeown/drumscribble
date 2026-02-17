import torch
import pytest
from drumscribble.export import trace_model, export_coreml
from drumscribble.model.drumscribble import DrumscribbleCNN


def test_trace_model():
    model = DrumscribbleCNN(
        backbone_dims=(32, 32, 32, 32),
        backbone_depths=(1, 1, 1, 1),
        num_attn_layers=1,
    )
    traced = trace_model(model, n_frames=200)
    # Verify traced model produces same shape
    x = torch.randn(1, 1, 128, 200)
    onset, vel, offset = traced(x)
    assert onset.shape == (1, 26, 200)


def _has_coremltools() -> bool:
    try:
        import coremltools  # noqa: F401

        return True
    except ImportError:
        return False


def _has_blob_writer() -> bool:
    """Check if coremltools BlobWriter is available (requires compiled C extension)."""
    try:
        from coremltools.converters.mil.backend.mil.load import BlobWriter

        return BlobWriter is not None
    except (ImportError, AttributeError):
        return False


@pytest.mark.skipif(not _has_coremltools(), reason="coremltools not installed")
@pytest.mark.skipif(
    not _has_blob_writer(),
    reason="coremltools BlobWriter unavailable (C extension not built for this Python)",
)
def test_export_coreml(tmp_path):
    model = DrumscribbleCNN(
        backbone_dims=(32, 32, 32, 32),
        backbone_depths=(1, 1, 1, 1),
        num_attn_layers=1,
    )
    path = tmp_path / "test.mlpackage"
    export_coreml(model, str(path), n_frames=200)
    assert path.exists()
