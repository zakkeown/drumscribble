import numpy as np


def _sine(freq=440.0, dur=1.0, sr=16000):
    """Generate a sine wave for testing."""
    t = np.arange(int(sr * dur)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestApplyRIR:
    def test_output_same_length(self):
        from scripts.hf_upload.audio_augment import apply_rir
        audio = _sine(dur=1.0)
        rir = np.zeros(8000, dtype=np.float32)
        rir[0] = 1.0  # identity RIR
        result = apply_rir(audio, rir, wet_mix=1.0)
        assert len(result) == len(audio)

    def test_identity_rir_preserves_signal(self):
        from scripts.hf_upload.audio_augment import apply_rir
        audio = _sine(dur=0.5)
        rir = np.zeros(100, dtype=np.float32)
        rir[0] = 1.0
        result = apply_rir(audio, rir, wet_mix=1.0)
        np.testing.assert_allclose(result, audio, atol=1e-5)

    def test_wet_dry_mix(self):
        from scripts.hf_upload.audio_augment import apply_rir
        audio = _sine(dur=0.5)
        rir = np.zeros(1600, dtype=np.float32)
        rir[0] = 1.0
        rir[800] = 0.5  # echo at 50ms
        dry_result = apply_rir(audio, rir, wet_mix=0.0)
        np.testing.assert_allclose(dry_result, audio, atol=1e-5)


class TestApplyEQ:
    def test_output_same_length(self):
        from scripts.hf_upload.audio_augment import apply_eq
        audio = _sine(dur=1.0)
        result = apply_eq(audio, sr=16000, low_shelf_db=3.0, high_shelf_db=-3.0,
                          mid_freq=1000.0, mid_db=2.0, mid_q=1.0)
        assert len(result) == len(audio)

    def test_zero_gains_preserve_signal(self):
        from scripts.hf_upload.audio_augment import apply_eq
        audio = _sine(dur=0.5)
        result = apply_eq(audio, sr=16000, low_shelf_db=0.0, high_shelf_db=0.0,
                          mid_freq=1000.0, mid_db=0.0, mid_q=1.0)
        np.testing.assert_allclose(result, audio, atol=1e-4)


class TestApplyNoise:
    def test_output_same_length(self):
        from scripts.hf_upload.audio_augment import apply_noise
        audio = _sine(dur=1.0)
        noise = np.random.randn(32000).astype(np.float32)
        result = apply_noise(audio, noise, snr_db=30.0)
        assert len(result) == len(audio)

    def test_high_snr_mostly_preserves_signal(self):
        from scripts.hf_upload.audio_augment import apply_noise
        audio = _sine(dur=0.5)
        noise = np.random.randn(8000).astype(np.float32) * 0.01
        result = apply_noise(audio, noise, snr_db=40.0)
        np.testing.assert_allclose(result, audio, atol=0.05)

    def test_noise_shorter_than_audio_loops(self):
        from scripts.hf_upload.audio_augment import apply_noise
        audio = _sine(dur=1.0, sr=16000)  # 16000 samples
        noise = np.random.randn(4000).astype(np.float32)  # 4000 samples
        result = apply_noise(audio, noise, snr_db=20.0)
        assert len(result) == len(audio)


class TestAugmentAudio:
    def test_returns_correct_count(self):
        from scripts.hf_upload.audio_augment import augment_audio
        audio = _sine(dur=1.0)
        rir = np.zeros(100, dtype=np.float32)
        rir[0] = 1.0
        noise = np.random.randn(16000).astype(np.float32)
        rng = np.random.default_rng(42)
        results = augment_audio(audio, sr=16000, rirs=[rir], noises=[noise],
                                n_copies=3, rng=rng)
        assert len(results) == 3
        for r in results:
            assert len(r) == len(audio)

    def test_each_copy_is_different(self):
        from scripts.hf_upload.audio_augment import augment_audio
        audio = _sine(dur=1.0)
        rir = np.zeros(1600, dtype=np.float32)
        rir[0] = 1.0
        rir[800] = 0.3
        noise = np.random.randn(16000).astype(np.float32)
        rng = np.random.default_rng(42)
        results = augment_audio(audio, sr=16000, rirs=[rir], noises=[noise],
                                n_copies=3, rng=rng)
        # At least some copies should differ
        assert not np.allclose(results[0], results[1], atol=1e-3)
