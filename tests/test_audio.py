import torch
from drumscribble.audio import compute_mel_spectrogram, load_and_preprocess
from drumscribble.config import SAMPLE_RATE, N_MELS, HOP_LENGTH


def test_mel_spectrogram_shape():
    waveform = torch.randn(1, SAMPLE_RATE * 10)  # 10s mono
    mel = compute_mel_spectrogram(waveform)
    assert mel.shape[0] == 1  # batch dim from channel
    assert mel.shape[1] == N_MELS  # 128 mel bins
    expected_frames = (SAMPLE_RATE * 10) // HOP_LENGTH + 1
    assert abs(mel.shape[2] - expected_frames) <= 1


def test_mel_spectrogram_log_scale():
    waveform = torch.randn(1, SAMPLE_RATE * 2)
    mel = compute_mel_spectrogram(waveform)
    # Log-mel should have reasonable values (not all zeros, not huge)
    assert mel.min() > -100
    assert mel.max() < 100


def test_mel_spectrogram_4d_output():
    waveform = torch.randn(1, SAMPLE_RATE * 5)
    mel = compute_mel_spectrogram(waveform, as_4d=True)
    # (1, 1, 128, T) for model input format
    assert mel.dim() == 4
    assert mel.shape[1] == 1
    assert mel.shape[2] == N_MELS


def test_load_and_preprocess_resamples(tmp_path):
    import torchaudio
    # Create a 48kHz test file
    waveform = torch.randn(2, 48000 * 2)  # 2s stereo at 48kHz
    path = tmp_path / "test.wav"
    torchaudio.save(str(path), waveform, 48000)

    audio = load_and_preprocess(str(path))
    assert audio.shape[0] == 1  # mono
    expected_samples = int(2 * SAMPLE_RATE)
    assert abs(audio.shape[1] - expected_samples) <= SAMPLE_RATE // 10
