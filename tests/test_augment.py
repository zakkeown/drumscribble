import torch
from drumscribble.data.augment import SpecAugment, AudioAugmentPipeline


def test_spec_augment_shape():
    aug = SpecAugment(freq_mask_param=10, time_mask_param=20, num_masks=2)
    mel = torch.randn(1, 128, 625)
    out = aug(mel)
    assert out.shape == mel.shape


def test_spec_augment_masks_something():
    torch.manual_seed(42)
    aug = SpecAugment(freq_mask_param=30, time_mask_param=50, num_masks=2)
    mel = torch.ones(1, 128, 625)
    out = aug(mel)
    assert (out == 0).any()


def test_audio_augment_pipeline():
    pipeline = AudioAugmentPipeline()
    waveform = torch.randn(1, 160000)  # 10s at 16kHz
    out = pipeline(waveform)
    assert out.shape == waveform.shape
