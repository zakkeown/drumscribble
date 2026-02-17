"""Audio preprocessing: loading, resampling, mel spectrogram."""
import torch
import torchaudio
from drumscribble.config import SAMPLE_RATE, N_MELS, HOP_LENGTH

_mel_transform = None


def _get_mel_transform(device: torch.device = torch.device("cpu")):
    global _mel_transform
    if _mel_transform is None or _mel_transform.mel_scale.fb.device != device:
        _mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=2048,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=20.0,
            f_max=8000.0,
            power=2.0,
        ).to(device)
    return _mel_transform


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    as_4d: bool = False,
) -> torch.Tensor:
    """Compute log-mel spectrogram from waveform.

    Args:
        waveform: (C, samples) or (samples,) tensor at SAMPLE_RATE.
        as_4d: If True, return (1, 1, N_MELS, T) for model input.

    Returns:
        Log-mel spectrogram: (C, N_MELS, T) or (1, 1, N_MELS, T).
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_spec = _get_mel_transform(waveform.device)(waveform)
    log_mel = torch.log(mel_spec.clamp(min=1e-7))

    if as_4d:
        if log_mel.shape[0] > 1:
            log_mel = log_mel.mean(dim=0, keepdim=True)
        return log_mel.unsqueeze(0)
    return log_mel


def load_and_preprocess(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio file, convert to mono, resample to target_sr.

    Returns:
        (1, samples) tensor at target_sr.
    """
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform
