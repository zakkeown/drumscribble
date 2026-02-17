"""Data augmentation for training."""
import torch
import torchaudio


class SpecAugment(torch.nn.Module):
    """SpecAugment: frequency and time masking on mel spectrograms.

    Applies random frequency and time masks to mel spectrograms during training.
    Based on: Park et al., "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition", 2019.

    Args:
        freq_mask_param: Maximum width of frequency masks.
        time_mask_param: Maximum width of time masks.
        num_masks: Number of frequency+time mask pairs to apply.
    """

    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 30,
        num_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param)
        self.num_masks = num_masks

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply frequency and time masks to a mel spectrogram.

        Args:
            mel: Mel spectrogram tensor of shape (batch, freq_bins, time_frames)
                or (channels, freq_bins, time_frames).

        Returns:
            Masked mel spectrogram with same shape as input.
        """
        for _ in range(self.num_masks):
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)
        return mel


class AudioAugmentPipeline(torch.nn.Module):
    """Waveform-level augmentation pipeline.

    Currently applies gain jitter (random volume scaling). Future additions
    (Phase 6): RIR convolution and pitch shift.

    Args:
        gain_range: Tuple of (min_gain, max_gain) for uniform random scaling.
    """

    def __init__(self, gain_range: tuple[float, float] = (0.5, 1.5)):
        super().__init__()
        self.gain_range = gain_range

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply gain jitter to a waveform.

        Args:
            waveform: Audio waveform tensor of shape (channels, samples).

        Returns:
            Gain-augmented waveform with same shape as input.
        """
        gain = torch.empty(1).uniform_(*self.gain_range).item()
        return waveform * gain
