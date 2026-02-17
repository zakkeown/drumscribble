"""CoreML export for DrumscribbleCNN."""
import torch
from drumscribble.model.drumscribble import DrumscribbleCNN
from drumscribble.config import N_MELS


def trace_model(model: DrumscribbleCNN, n_frames: int = 625) -> torch.jit.ScriptModule:
    """Trace model for export. Returns TorchScript module."""
    model.eval()
    example = torch.randn(1, 1, N_MELS, n_frames)
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
    return traced


def export_coreml(
    model: DrumscribbleCNN,
    output_path: str,
    n_frames: int = 625,
) -> None:
    """Export model to CoreML mlpackage format.

    Exports a fixed-shape model. Call multiple times with different
    n_frames for 10s/20s/30s variants (Revision 5).
    """
    import coremltools as ct

    traced = trace_model(model, n_frames)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="mel_spectrogram", shape=(1, 1, N_MELS, n_frames))],
        outputs=[
            ct.TensorType(name="onset_probs"),
            ct.TensorType(name="velocity"),
            ct.TensorType(name="offset_probs"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
    )

    mlmodel.save(output_path)
