"""Waveform augmentation for offline dataset generation.

Pure numpy/scipy — no torch dependency. Designed for use in compute_features_aug.py.
"""

import numpy as np
from scipy.signal import fftconvolve, sosfilt


def apply_rir(
    audio: np.ndarray,
    rir: np.ndarray,
    wet_mix: float = 0.7,
) -> np.ndarray:
    """Convolve audio with a room impulse response.

    Args:
        audio: 1-D float32 audio signal.
        rir: 1-D float32 impulse response.
        wet_mix: Blend factor (0.0 = dry, 1.0 = fully wet).

    Returns:
        Blended audio, same length as input.
    """
    wet = fftconvolve(audio, rir, mode="full")[: len(audio)]
    # Normalize wet to match dry RMS to prevent volume jumps
    dry_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
    wet_rms = np.sqrt(np.mean(wet ** 2)) + 1e-8
    wet = wet * (dry_rms / wet_rms)
    return ((1.0 - wet_mix) * audio + wet_mix * wet).astype(np.float32)


def _low_shelf_sos(freq: float, gain_db: float, sr: int) -> np.ndarray:
    """Design a low shelf biquad filter as second-order sections."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * 0.707)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _high_shelf_sos(freq: float, gain_db: float, sr: int) -> np.ndarray:
    """Design a high shelf biquad filter as second-order sections."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * 0.707)
    cos_w0 = np.cos(w0)
    sqrt_A = np.sqrt(A)
    b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
    a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
    a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
    a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _peak_sos(freq: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    """Design a peak/notch biquad filter as second-order sections."""
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cos_w0 = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cos_w0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w0
    a2 = 1 - alpha / A
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def apply_eq(
    audio: np.ndarray,
    sr: int,
    low_shelf_db: float = 0.0,
    high_shelf_db: float = 0.0,
    mid_freq: float = 1000.0,
    mid_db: float = 0.0,
    mid_q: float = 1.0,
    low_shelf_freq: float = 150.0,
    high_shelf_freq: float = 6000.0,
) -> np.ndarray:
    """Apply parametric EQ (low shelf + mid peak + high shelf).

    All gains in dB. Zero gain = passthrough.
    """
    sos_list = []
    if abs(low_shelf_db) > 0.01:
        sos_list.append(_low_shelf_sos(low_shelf_freq, low_shelf_db, sr))
    if abs(mid_db) > 0.01:
        sos_list.append(_peak_sos(mid_freq, mid_db, mid_q, sr))
    if abs(high_shelf_db) > 0.01:
        sos_list.append(_high_shelf_sos(high_shelf_freq, high_shelf_db, sr))
    if not sos_list:
        return audio.copy()
    sos = np.concatenate(sos_list, axis=0)
    return sosfilt(sos, audio).astype(np.float32)


def apply_noise(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 30.0,
) -> np.ndarray:
    """Add noise at a target SNR.

    Noise is looped if shorter than audio, trimmed if longer.
    """
    if len(noise) < len(audio):
        repeats = (len(audio) // len(noise)) + 1
        noise = np.tile(noise, repeats)
    noise = noise[: len(audio)]
    sig_power = np.mean(audio ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return (audio + scale * noise).astype(np.float32)


def augment_audio(
    audio: np.ndarray,
    sr: int,
    rirs: list[np.ndarray],
    noises: list[np.ndarray],
    n_copies: int = 3,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Generate n_copies augmented variants of an audio signal."""
    if rng is None:
        rng = np.random.default_rng()
    results = []
    for _ in range(n_copies):
        aug = audio.copy()
        rir = rirs[rng.integers(len(rirs))]
        wet_mix = rng.uniform(0.3, 0.9)
        aug = apply_rir(aug, rir, wet_mix=wet_mix)
        aug = apply_eq(
            aug, sr=sr,
            low_shelf_db=rng.uniform(-6.0, 6.0),
            high_shelf_db=rng.uniform(-6.0, 6.0),
            low_shelf_freq=rng.uniform(80.0, 200.0),
            high_shelf_freq=rng.uniform(4000.0, 8000.0),
            mid_freq=rng.uniform(300.0, 3000.0),
            mid_db=rng.uniform(-4.0, 4.0),
            mid_q=rng.uniform(0.7, 2.0),
        )
        noise = noises[rng.integers(len(noises))]
        snr_db = rng.uniform(20.0, 40.0)
        aug = apply_noise(aug, noise, snr_db=snr_db)
        results.append(aug)
    return results
