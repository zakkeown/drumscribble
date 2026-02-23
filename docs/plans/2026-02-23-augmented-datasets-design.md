# Augmented Dataset Generation Design

**Date:** 2026-02-23
**Status:** Approved

## Problem

DrumscribbleCNN was trained on electronic drums (E-GMD) and synthesized stems (STAR/StemGMD). Evaluation on MDB-Drums (real acoustic drums, unseen) shows the model has near-perfect precision (99%) but only 11% recall at threshold=0.5. The model doesn't recognize acoustic drum hits because they sound fundamentally different from the dry electronic signals it trained on.

The three biggest domain gaps: room reverb, frequency response variation, and background noise floor. Electronic drums are perfectly dry, spectrally consistent, and silent between hits. Real recordings are none of these.

## Approach

Offline augmented dataset generation. Extend the existing `compute_features.py` pipeline with a waveform augmentation stage inserted between audio loading and mel spectrogram computation. For each original track, produce 3 augmented variants plus keep the dry original (4x total data). Output as Parquet shards in the identical format used by the training pipeline.

This means the training loop changes zero lines of code — just point at the new dataset.

## Augmentation Chain

Each augmented variant applies three transforms in sequence:

```
dry audio → RIR convolution → parametric EQ → noise injection → compute mel
```

### 1. Room Impulse Response Convolution

- **Source:** OpenSLR SLR28 (simulated RIRs — thousands of WAVs, already 16kHz matching our SAMPLE_RATE; real RIRs from 3 rooms of different sizes)
- **Method:** `scipy.signal.fftconvolve(dry, rir, mode="full")[:len(dry)]`
- **Wet/dry mix:** Random uniform 0.3–0.9 (always preserve some direct signal)
- **Skip dEchorate:** 48kHz HDF5 with multi-channel complexity, marginal gain over OpenSLR's variety

### 2. Parametric EQ

Three randomized biquad filters applied via `scipy.signal.sosfilt`:

| Band | Frequency | Gain | Q |
|------|-----------|------|---|
| Low shelf | 80–200 Hz | ±6 dB | fixed 0.707 |
| Mid peak | 300–3000 Hz | ±4 dB | 0.7–2.0 |
| High shelf | 4–8 kHz | ±6 dB | fixed 0.707 |

Simulates mic choice, placement, and room coloration.

### 3. Background Noise

- **Source:** OpenSLR point-source noise recordings (real ambient environments)
- **Fallback:** Generated pink noise for additional variety
- **SNR:** 20–40 dB (subtle — real recordings have hiss/room tone but aren't noisy)
- **Method:** Loop/trim noise to match audio length, scale to target SNR

### Annotation Invariance

None of these transforms shift onset times. The original MIDI/TSV annotations apply identically to augmented audio. Onset targets, velocity targets, and all metadata carry through unchanged.

## Output Datasets

| Dataset | Tracks (orig) | Tracks (augmented) | Est. Parquet size |
|---------|---------------|--------------------|-------------------|
| `schismaudio/e-gmd-aug` | ~49K | ~196K | ~400 GB |
| `schismaudio/star-drums-aug` | ~6K | ~24K | ~50 GB |

### Parquet Schema

Identical to existing feature datasets plus one new field:

| Field | Type | Description |
|-------|------|-------------|
| `augmentation` | string | JSON describing transforms applied (empty for dry originals) |

All existing fields preserved: mel_spectrogram, onset_targets, velocity_targets, n_frames, n_mels, n_classes, sample_rate, hop_length, fps, duration, split, and dataset-specific metadata.

## HF Jobs Execution

One PEP 723 UV script per dataset (`scripts/hf_upload/compute_features_aug.py`). CPU-only — no GPU needed for audio DSP.

**Instance sizing:** Use a generously provisioned instance. The cost of crashing at hour 20 and re-running from scratch far exceeds the marginal cost of a bigger machine. E-GMD raw audio is ~90 GB, augmented Parquet output will be ~400 GB, plus RIRs and working memory. Budget for at least 100 GB RAM and 500+ GB disk.

```
hf jobs uv run --flavor cpu-xxlarge --timeout 24h -s HF_TOKEN \
    scripts/hf_upload/compute_features_aug.py \
    --dataset egmd --augment-copies 3
```

Pipeline per job:

1. Download raw dataset (E-GMD WAV + MIDI from original source, or STAR FLAC + TSV)
2. Download OpenSLR RIRs + noises from `schismaudio/openslr-rirs` on HF Hub
3. For each track: load audio → produce 3 augmented + 1 dry → compute mel for each → accumulate rows
4. Write Parquet shards (500 MB max, zstd compression) with periodic flushing
5. Upload shards to HF Hub via `huggingface_hub.upload_large_folder`

Memory management: process tracks sequentially, flush completed shards to disk immediately, upload in batches. Same patterns that worked for the original feature computation. Aggressive shard flushing — don't accumulate rows in memory across the entire dataset.

## Training Integration

After augmented datasets are uploaded:

```
hf jobs uv run --flavor a10g-large --timeout 24h -s HF_TOKEN \
    scripts/train_hf_job.py \
    --dataset multi --epochs 100 --resume-from final.pt
```

The `--dataset multi` flag already supports loading from multiple Parquet feature datasets. Point the shard discovery at the augmented repos instead of (or in addition to) the originals.

## MDB-Drums Evaluation Baseline

Current results on MDB-Drums (23 tracks, completely unseen acoustic drums):

| Threshold | 5-class F1 | Precision | Recall |
|-----------|------------|-----------|--------|
| 0.5 | 0.197 | 0.992 | 0.109 |
| 0.3 | 0.434 | 0.955 | 0.281 |
| 0.2 | 0.597 | 0.865 | 0.456 |
| 0.1 | 0.628 | 0.481 | 0.904 |

Target after fine-tuning on augmented data: 5-class F1 > 0.7 at threshold 0.3–0.5.
