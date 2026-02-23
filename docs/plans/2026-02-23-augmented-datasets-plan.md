# Augmented Dataset Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Produce `schismaudio/e-gmd-aug` and `schismaudio/star-drums-aug` HF datasets with room reverb, EQ variation, and background noise augmentation applied to the original dry audio before mel spectrogram computation.

**Architecture:** A numpy/scipy augmentation module provides three transforms (RIR convolution, parametric EQ, noise injection). A PEP 723 UV script extends the existing `compute_features.py` pipeline, inserting augmentation between audio loading and mel computation. Each original track produces 3 augmented + 1 dry variant. Completed Parquet shards are stream-uploaded to HF Hub and deleted locally to manage disk.

**Tech Stack:** numpy, scipy (fftconvolve, sosfilt, iirpeak, shelving filters), librosa, soundfile, pyarrow, huggingface_hub. HF Jobs on L4x1 ($0.80/hr, 400 GB disk).

---

### Task 1: Audio Augmentation Module

**Files:**
- Create: `scripts/hf_upload/audio_augment.py`
- Test: `tests/test_audio_augment.py`

This module contains pure numpy/scipy functions for waveform augmentation. It has NO torch dependency (compute_features.py uses numpy/librosa, not torch).

**Step 1: Write failing tests**

Create `tests/test_audio_augment.py`:

```python
import numpy as np
import pytest


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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_audio_augment.py -v`
Expected: ImportError — module does not exist yet

**Step 3: Write the augmentation module**

Create `scripts/hf_upload/audio_augment.py`:

```python
"""Waveform augmentation for offline dataset generation.

Pure numpy/scipy — no torch dependency. Designed for use in compute_features_aug.py.
"""

import numpy as np
from scipy.signal import fftconvolve, sosfilt, iirpeak
from scipy.signal.filter_design import butter


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
    dry_rms = np.sqrt(np.mean(audio ** 2) + 1e-8)
    wet_rms = np.sqrt(np.mean(wet ** 2) + 1e-8)
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
    # Loop noise to match audio length
    if len(noise) < len(audio):
        repeats = (len(audio) // len(noise)) + 1
        noise = np.tile(noise, repeats)
    noise = noise[: len(audio)]

    # Scale noise to target SNR
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
    """Generate n_copies augmented variants of an audio signal.

    Each copy gets a random RIR, random EQ curve, and random noise.

    Args:
        audio: 1-D float32 audio.
        sr: Sample rate.
        rirs: List of 1-D RIR arrays to sample from.
        noises: List of 1-D noise arrays to sample from.
        n_copies: Number of augmented copies to produce.
        rng: Numpy random generator for reproducibility.

    Returns:
        List of n_copies augmented audio arrays.
    """
    if rng is None:
        rng = np.random.default_rng()

    results = []
    for _ in range(n_copies):
        aug = audio.copy()

        # 1. RIR convolution
        rir = rirs[rng.integers(len(rirs))]
        wet_mix = rng.uniform(0.3, 0.9)
        aug = apply_rir(aug, rir, wet_mix=wet_mix)

        # 2. Parametric EQ
        aug = apply_eq(
            aug,
            sr=sr,
            low_shelf_db=rng.uniform(-6.0, 6.0),
            high_shelf_db=rng.uniform(-6.0, 6.0),
            low_shelf_freq=rng.uniform(80.0, 200.0),
            high_shelf_freq=rng.uniform(4000.0, 8000.0),
            mid_freq=rng.uniform(300.0, 3000.0),
            mid_db=rng.uniform(-4.0, 4.0),
            mid_q=rng.uniform(0.7, 2.0),
        )

        # 3. Noise injection
        noise = noises[rng.integers(len(noises))]
        snr_db = rng.uniform(20.0, 40.0)
        aug = apply_noise(aug, noise, snr_db=snr_db)

        results.append(aug)

    return results
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_audio_augment.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add scripts/hf_upload/audio_augment.py tests/test_audio_augment.py
git commit -m "feat: add numpy/scipy audio augmentation module (RIR, EQ, noise)"
```

---

### Task 2: Augmented Feature Computation Script

**Files:**
- Create: `scripts/hf_upload/compute_features_aug.py`

This is a PEP 723 UV script for HF Jobs that extends the existing `compute_features.py` with augmentation. It downloads the raw dataset from HF Hub, downloads OpenSLR RIRs, processes all tracks with augmentation, writes Parquet shards, and stream-uploads them.

**Step 1: Write the script**

Create `scripts/hf_upload/compute_features_aug.py`. This is a standalone PEP 723 script (all dependencies declared inline). Key design:

- Reuses all feature computation logic from `compute_features.py` (copy the essential functions to keep it self-contained for HF Jobs)
- Adds augmentation stage between `load_audio_mono()` and `compute_mel()`
- Stream-uploads completed Parquet shards to HF Hub (upload then delete locally)
- Logs progress with RSS monitoring (same pattern as `train_hf_job.py`)
- Supports both `egmd` and `star` datasets via `--dataset` flag
- `--augment-copies N` controls how many augmented variants per track (default 3)
- `--seed` for reproducibility
- Processes train split only (val/test stay unaugmented for clean evaluation)

The PEP 723 header should declare: numpy, scipy, librosa, soundfile, pretty-midi, pyarrow, huggingface-hub.

The script structure:

```
1. Parse args (--dataset, --augment-copies, --seed, --output-repo)
2. Download raw dataset from HF Hub via snapshot_download
3. Download OpenSLR RIRs from schismaudio/openslr-rirs via snapshot_download
4. Load all RIR WAV files into memory (they're small, ~few hundred KB each)
5. Load all noise WAV files into memory
6. Discover entries (reuse egmd/star discovery logic)
7. For each entry:
   a. Load audio
   b. Generate dry + N augmented copies
   c. For each copy: compute mel, compute onset targets, build row dict
   d. Append rows to current shard buffer
   e. When buffer > 500MB: write Parquet shard, upload, delete local file
8. Flush final shard
9. Upload dataset card README
```

**Step 2: Smoke test locally**

Before running on HF Jobs, test with a subset:

```bash
uv run python scripts/hf_upload/compute_features_aug.py \
    --dataset egmd --augment-copies 2 --max-entries 10 \
    --output-dir /tmp/aug-test --no-upload
```

Verify:
- Output Parquet files exist in `/tmp/aug-test/`
- Each original entry produced 3 rows (1 dry + 2 augmented)
- Mel spectrogram shapes are correct (128, N_FRAMES)
- Augmentation metadata field is populated

**Step 3: Commit**

```bash
git add scripts/hf_upload/compute_features_aug.py
git commit -m "feat: add augmented feature computation script for HF Jobs"
```

---

### Task 3: Launch E-GMD Augmentation Job

**Files:**
- None (operational task)

**Step 1: Create the HF dataset repo**

```bash
huggingface-cli repo create schismaudio/e-gmd-aug --type dataset
```

**Step 2: Push the script and launch**

```bash
hf jobs uv run --flavor l4x1 --timeout 24h -s HF_TOKEN \
    scripts/hf_upload/compute_features_aug.py \
    --dataset egmd --augment-copies 3 --output-repo schismaudio/e-gmd-aug
```

**Step 3: Monitor**

```bash
hf jobs logs <JOB_ID> --timeout 60
hf jobs inspect <JOB_ID>
```

Check for:
- RIR download completes
- E-GMD download completes
- Tracks processing with augmentation (progress logs)
- Shard uploads succeeding
- RSS staying stable (no memory leak)

---

### Task 4: Launch STAR Augmentation Job

**Files:**
- None (operational task)

**Step 1: Create the HF dataset repo**

```bash
huggingface-cli repo create schismaudio/star-drums-aug --type dataset
```

**Step 2: Launch**

STAR is much smaller than E-GMD, so a lighter instance works:

```bash
hf jobs uv run --flavor l4x1 --timeout 12h -s HF_TOKEN \
    scripts/hf_upload/compute_features_aug.py \
    --dataset star --augment-copies 3 --output-repo schismaudio/star-drums-aug
```

**Step 3: Monitor**

Same as Task 3.

---

### Task 5: Verify and Update Training Config

**Files:**
- Modify: `scripts/train_hf_job.py` (shard discovery to include augmented repos)

**Step 1: Verify augmented datasets load correctly**

```python
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

path = snapshot_download("schismaudio/e-gmd-aug", repo_type="dataset")
table = pq.read_table(f"{path}/train-00000-of-00XXX.parquet")
print(table.schema)
print(table.num_rows)
print(table.column("augmentation")[0].as_py())
```

**Step 2: Update training script shard discovery**

Add the augmented dataset repos to the multi-dataset shard discovery in `train_hf_job.py`. The augmented shards use the identical schema so no loader changes are needed — just add the repo IDs to the download list.

**Step 3: Commit**

```bash
git add scripts/train_hf_job.py
git commit -m "feat: add augmented dataset repos to training shard discovery"
```

---

### Hardware Decision Reference

| Flavor | Disk | RAM | Cost/hr | E-GMD 24h | Rationale |
|--------|------|-----|---------|-----------|-----------|
| cpu-xl | 50 GB | ? | ~$0.05 | ~$1.20 | NOT ENOUGH DISK |
| L4x1 | 400 GB | 30 GB | $0.80 | $19.20 | **Best value** — 400GB disk fits E-GMD + stream upload |
| A10G-large | 200 GB | 46 GB | $1.50 | $36.00 | Tight on disk but more RAM |
| 2xA10G-large | 1000 GB | 92 GB | $3.00 | $72.00 | Overkill but safe |

**Recommendation:** L4x1 for both jobs. Stream-upload completed shards so peak disk = raw dataset + buffer.
