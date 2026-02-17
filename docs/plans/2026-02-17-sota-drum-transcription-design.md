# SOTA On-Device CoreML Drum Transcription — Design Document

**Date**: 2026-02-17
**Status**: Reviewed (5 iterative passes)
**Target**: Research-grade accuracy, CoreML/ANE deployment on M4/A18

---

## 1. Model Architecture: DrumscribbleCNN

### Overview

U-Net encoder-decoder with ConvNeXt backbone, attention bottleneck, and skip connections, optimized for Apple Neural Engine (ANE). Approximately 12.1M parameters.

### ConvNeXt Backbone

4 stages with increasing channel depth:

| Stage | Blocks | Channels | Kernel | Downsampling |
|-------|--------|----------|--------|-------------|
| 1 | 5 | 64 | (1,7) | Stride-2 Conv |
| 2 | 5 | 128 | (1,7) | Stride-2 Conv |
| 3 | 5 | 256 | (1,11)* | Stride-2 Conv |
| 4 | 5 | 384 | (1,11)* | None |

*\*Revision 1: Use (1,11) asymmetric temporal kernels instead of square 11×11. Temporal patterns dominate in drum transcription. If (1,11) falls back from ANE, decompose into InceptionNeXt-style parallel branches: (1,3) + (1,7) + identity.*

Each ConvNeXt block:
```
Input → Depthwise Conv(kernel) → BatchNorm → 1×1 Conv (expand 4×) → GELU → 1×1 Conv (project back) → Residual Add
```

Channel dimensions must be **multiples of 32** for ANE alignment.

Each of Stages 1-3 provides a **skip connection** to the U-Net decoder at resolutions T, T/2, T/4 respectively. Stage 4 outputs to the attention bottleneck at T/8.

### Self-Attention Layers (3 layers, at T/8 bottleneck)

Following Apple's ml-ane-transformers pattern exactly (Revision 9):
- All Linear layers → Conv2d(1×1) for ANE
- Tensor format: (B, C, 1, S) — ANE's native 4D format (Revision 8)
- Q·K attention via `torch.einsum('bchq,bkhc->bkhq')`
- 4 attention heads, 384 channels
- For 30s chunks (1875 frames): consider block-diagonal attention (2×~938 blocks) for ~65% speedup per WhisperKit's approach

### U-Net Decoder (3 upsampling stages)

The 3 temporal downsamplings in the encoder reduce temporal resolution to T/8 (efficient for attention), but the output heads require full T resolution. A U-Net decoder with skip connections recovers resolution:

| Stage | Input | Skip | Output | Operation |
|-------|-------|------|--------|-----------|
| Up 1 | 384, T/8 | 256, T/4 | 256, T/4 | Interpolate → concat → Conv(1×1) fuse → ConvNeXt block |
| Up 2 | 256, T/4 | 128, T/2 | 128, T/2 | Interpolate → concat → Conv(1×1) fuse → ConvNeXt block |
| Up 3 | 128, T/2 | 64, T | 64, T | Interpolate → concat → Conv(1×1) fuse → ConvNeXt block |

**Rationale**: T/8 = 128ms per frame, which exceeds the 50ms onset tolerance for mir_eval. Skip connections carry fine-grained temporal information from the encoder stages directly to the decoder, preserving onset precision. This is the same pattern used in Apple's Stable Diffusion ANE implementation and standard audio segmentation models. All operations (interpolate, concat, Conv2d 1×1, ConvNeXt blocks) are ANE-native.

**Parameter cost**: ~0.92M for the decoder (trivial vs. ~9.4M encoder).

### Output Heads (3 heads, multi-output)

1. **Onset logits** → Sigmoid → (B, 26, T) onset probability per class per frame
2. **Velocity** → Sigmoid → (B, 26, T) velocity [0,1] per class per frame
3. **Offset logits** → Sigmoid → (B, 26, T) offset probability per class per frame

Return as plain tuple for clean `torch.jit.trace` conversion (no dict/NamedTuple).

### Normalization: BatchNorm (Frozen for Local Training)

BatchNorm is the only ANE-compatible normalization. It fuses with Conv during CoreML conversion.

**Frozen BatchNorm Protocol (Revision 3)**:
- Cloud training (A100, batch 32+): Normal BatchNorm with running statistics
- Local training (MPS, batch 4-8): Freeze BN layers to eval mode, use running statistics from cloud checkpoint
- Sync BN stats every N cloud epochs

### Activation: GELU

GELU is the ConvNeXt standard. Supported in MLProgram format (Core ML 5+, iOS 15+). Monitor for float16 precision drift on ANE — if issues arise, decompose into simpler ops.

---

## 2. Input Representation & MERT Integration

### Dual-Mode Design with FiLM Conditioning

**Primary input**: Log-mel spectrogram (128 bins, 16kHz, 10ms hop → 62.5fps)

**Optional input**: MERT-95M features from **layer 5-6** (Revision 2). Research confirmed 95M is better than 330M for beat tracking (88.3 vs 87.9 F1). Layer 5-6 of 95M corresponds to layer 10 of 330M.

**FiLM (Feature-wise Linear Modulation)** conditioning:
- `y = gamma * x + beta` where gamma, beta are projected from MERT features
- **Implementation (Revision 6)**: Pre-expand gamma/beta tensors to match input shape before multiply/add — no broadcasting at CoreML level. Alternatively, implement as 1×1 depthwise conv + bias.
- Training dropout: 15% complete MERT dropout, 50% partial (random layer subset)
- Model must work **with or without** MERT features at inference

### Three-Tier Inference Pipeline (Revision 7)

1. **Swift/Accelerate**: Raw audio → log-mel spectrogram (vDSP, float32). Cannot include FFT/STFT in CoreML — no native support.
2. **MERT CoreML model** (optional, separate): mel → MERT features (~190MB for 95M)
3. **DrumscribbleCNN CoreML model**: features → onset/velocity/offset

---

## 3. Output Format & Target Representation

### 26 GM-Standard Drum Classes

```
35: Acoustic Bass Drum    36: Electric Bass Drum    37: Side Stick
38: Acoustic Snare        39: Hand Clap             40: Electric Snare
41: Low Floor Tom         42: Closed Hi-Hat         43: High Floor Tom
44: Pedal Hi-Hat          45: Low Tom               46: Open Hi-Hat
47: Low-Mid Tom           48: Hi-Mid Tom            49: Crash Cymbal 1
50: High Tom              51: Ride Cymbal 1         52: Chinese Cymbal
53: Ride Bell             54: Tambourine            55: Splash Cymbal
56: Cowbell               57: Crash Cymbal 2        59: Ride Cymbal 2
75: Claves                76: Hi Wood Block         77: Low Wood Block
```

### Target Widening (Published Standard)

Soft targets: [0.3, 0.6, 1.0, 0.6, 0.3] pyramid centered on onset frame. Used by ADTOF v2 (2023) and STAR Drums (2024).

Note: OaF Drums chose single-frame hard targets for drum-specific work. The optimal width may depend on architecture and frame resolution — include in ablation experiments.

### Class Mappings for Evaluation

Document exact 26→N mappings for each benchmark (Revision 16):
- E-GMD: 26→7 (kick, snare, hi-hat open, hi-hat closed, tom, crash, ride+bell)
- MDB: 26→5 (kick, snare, tom, hi-hat, cymbal)
- IDMT: 26→3 (kick, snare, hi-hat)
- STAR: 26→18 (direct subset mapping, all 18 STAR classes have GM equivalents)

---

## 4. Loss Functions

### Default: BCE + Target Widening (Revision 11)

Every published ADT system uses plain Binary Cross-Entropy. ASL has never been validated for drum transcription. The target widening approach already addresses class imbalance by spreading the positive signal across more frames.

```
L_onset = BCE(onset_pred, soft_target)
L_velocity = masked_MSE(vel_pred, vel_target) * 0.5   # only where onset > 0
L_offset = BCE(offset_pred, soft_target)
L_total = L_onset + L_velocity + L_offset
```

### Ablation: Asymmetric Focal Loss

Test as experiment, not default. If ablating:
- γ+=0 (never suppress rare positives), γ- in {1, 2, 4}, clip m=0.05
- Note: target widening and ASL may be partially redundant — both address class imbalance through different mechanisms

### Velocity Loss

Masked MSE at 0.5 weight, applied only at onset positions. N2N's Annealed Pseudo-Huber loss is an alternative worth testing (transitions MSE→MAE during training).

---

## 5. Training Pipeline & Data Strategy

### Stage 1: Synthetic (E-GMD Re-synthesis)

**Goal**: Diverse timbral coverage with clean MIDI labels.

- Render E-GMD MIDI with 10-13 acoustic SFZ drum kits via **sfizz** (SFZ) and **FluidSynth** (SF2)
- Available kits: Salamander, SM Drums, Naked Drums, Big Rusty Drums, Swirly Drums, Unruly Drums, DRS Kit, Muldjord Kit, and others from sfzinstruments.github.io
- Augmentation chains (pitch shift, EQ, saturation) → 42-56 virtual variants (Revision 4)
- RIR convolution: 1,600+ real impulse responses from MIT IR Survey + BUT ReverbDB + EchoThief (cost: <1ms/batch on GPU)
- **Rendering time**: 2-4 days on M4 Max (parallelized across 8 cores), one-time cost
- **Storage**: ~200-400GB FLAC output

### Stage 2: Real Data (E-GMD + STAR)

**Goal**: Bridge synthetic-to-real gap with human-performed, real-world audio.

- E-GMD original audio: 444 hours, isolated drums, full GM MIDI + velocity
- STAR Drums: 100.6 hours training, 18 classes, real accompaniment
- **Preprocessing (Revision 14)**: Run `mix_musdb_items.py` for STAR. Resample 48kHz→16kHz. Parse tab-separated annotations to frame-level targets at 62.5fps.
- **Mix-training (Revision 15)**: 50% isolated drums, 50% mixed with random non-drum stems from MUSDB18-HQ
- Loss: BCE + target widening

### Stage 3: MERT-Conditioned Fine-Tuning

- Enable MERT feature extraction
- FiLM conditioning with dropout schedule
- Lower learning rate for backbone, higher for FiLM projection layers

### ADTOF Role (Revision 12)

ADTOF provides only 5-class labels and pre-computed spectrograms. Options:
- Use as 5-class auxiliary supervision with masked loss
- Use for pre-training only
- Skip entirely — E-GMD + STAR provide 544h of fine-grained data

### Training Infrastructure

| Environment | Use Case | Batch Size | Estimated Speed |
|-------------|----------|------------|-----------------|
| M4 Max (MPS) | Dev, debug, overfitting tests, validation | 4-8 | ~2-5 it/s |
| A100 80GB (HF Jobs) | Full training runs | 32 | ~25-50 it/s |

**Estimated cloud cost**: ~$50-100 total on HF Jobs ($2.50/hr A100)

---

## 6. Post-Processing & Inference

### Peak Picking

Per-class threshold calibration via grid search on validation set (never test set). Peak picking with 30ms pre / 100ms post context. 30ms NMS (needs per-class tuning — hi-hat requires tighter NMS than crash cymbal).

### Windowed Inference

10-30 second chunks with 2-5 second overlap. Overlap-add stitching at boundaries.

**Fixed-shape models (Revision 5)**: Export 3 separate CoreML models for 10s/20s/30s instead of EnumeratedShapes. Confirmed regression in `mlprogram` causes non-default shapes to fall back from ANE (10-19x slower).

---

## 7. CoreML Deployment & ANE Optimization

### Conversion

```python
import coremltools as ct  # 8.2+ required (Revision 10)

traced = torch.jit.trace(model, example_input)
mlmodel = ct.convert(
    traced,
    inputs=[ct.TensorType(shape=(1, 128, 1, T))],  # (B, C, 1, T) 4D format
    outputs=[
        ct.TensorType(name="onset_logits"),
        ct.TensorType(name="velocity"),
        ct.TensorType(name="offset_logits"),
    ],
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS17,
)
```

Use coremltools 8.2+ for `scaled_dot_product_attention_sliced_q` graph pass (34% faster, 45% less memory for long-sequence attention).

### Quantization

6-bit palettization: ~2.5× size reduction (24-36MB → 9-14MB). Negligible latency gain for models this small. Use Sensitive K-Means (calibration-data-based) for best accuracy. Validate on evaluation set before deploying.

### ANE Constraints Summary

| Constraint | Status |
|------------|--------|
| BatchNorm | Safe — fuses with Conv |
| GELU | Supported (MLProgram, iOS 15+) — monitor float16 |
| Sigmoid | Safe — fully supported |
| Depthwise Conv (1,7) | Proven (FastViT) |
| Depthwise Conv (1,11) | Unknown — needs 30-min empirical test |
| Self-attention at seq 625-1875 | Proven (WhisperKit 1500, Depth-Anything 1814) |
| FiLM (element-wise mul+add) | Medium risk — pre-expand tensors to avoid broadcast fallback |
| No RNN/LSTM/GRU | OK — pure CNN + attention |
| No dilated convolutions | OK — standard stride-1 convolutions |
| Channels multiples of 32 | OK — 64/128/256/384 |

---

## 8. Training Hyperparameters & Tooling

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-3 (backbone), 3e-3 (heads) |
| Weight decay | 0.05 |
| LR schedule | Cosine with warmup (5% of total steps) |
| Batch size | 32 (A100) / 4-8 (MPS) |
| Mixed precision | AMP bf16 (A100) / fp32 (MPS) |
| Gradient clipping | 1.0 |
| EMA | 0.999 decay |

### Augmentation Pipeline

| Augmentation | Parameters | Applied |
|-------------|------------|---------|
| RIR convolution | 1,600+ real IRs | Stage 1-2 |
| Pitch shift | ±2 semitones | Stage 1-2 |
| Time stretch | ±5% | Stage 1 |
| EQ (parametric) | Random curves | Stage 1-2 |
| Compression/saturation | Light-medium | Stage 1 |
| Accompaniment mixing | MUSDB18-HQ stems | Stage 2 |
| SpecAugment | 2 time, 2 freq masks | All stages |

### Tooling

- **sfizz**: SFZ soundfont rendering for Stage 1 synthesis
- **FluidSynth**: SF2 soundfont rendering
- **HuggingFace Jobs**: A100 cloud training ($2.50/hr)
- **Weights & Biases / Trackio**: Experiment tracking
- **coremltools 8.2+**: CoreML conversion with sliced-Q attention
- **Xcode Instruments**: ANE profiling and verification
- **mir_eval**: Evaluation metrics

---

## 9. Evaluation Protocol (Revision 16)

### Metrics

- **Onset F1**: `mir_eval.transcription.onset_precision_recall_f1` at 50ms tolerance
- **Velocity F1**: `mir_eval.transcription_velocity.precision_recall_f1_overlap` at 0.1 normalized tolerance
- Report per-class F1 + overall/sum F1 for each dataset
- Threshold optimization on validation set only

### Benchmarks and Targets (Revision 17)

| Dataset | Classes | Current SOTA | Source | Our Target |
|---------|---------|-------------|--------|------------|
| E-GMD (onset) | 7 | 89.68 | N2N 10-step | >90.0 |
| E-GMD (velocity) | 7 | 82.80 | N2N 10-step | >83.0 |
| MDB Drums | 5 | 87.86 | N2N 10-step | >88.0 |
| IDMT-SMT | 3 | 94.90 | N2N 10-step | >95.0 |
| ENST Drums | 5 | ~85 | Riley 2025 | >85.0 |
| STAR Drums | 18 | ~70 | Dynamic FSL | >70.0 |

**Realistic first-run targets** (CNN + MERT, no diffusion): E-GMD ~87, MDB ~82 (based on N2N MFM-only ablation).

### Key Comparability Notes

- IDMT is near saturation (~95%). Marginal gains are not meaningful.
- N2N trains only on E-GMD. If we train on STAR/ADTOF too, MDB/IDMT scores are not directly comparable.
- DTD (drum-only) vs DTM (drum-in-mix) must be stated for each evaluation.
- Velocity evaluation only possible on E-GMD (no other dataset has velocity ground truth).

---

## 10. Open Questions (Require Empirical Testing)

1. **Kernel (1,11) on ANE**: 30-minute spike test with Xcode Instruments. If fallback, use InceptionNeXt decomposition.
2. **ReLU vs GELU on ANE**: Accuracy impact unknown. GELU is ConvNeXt standard; ReLU is safer.
3. **Block-diagonal attention for 30s chunks**: WhisperKit showed 65% speedup. Worth testing.
4. **BCE vs ASL ablation**: Run early in training on E-GMD subset.
5. **NMS interval per-class**: Hi-hat needs tighter NMS than crash cymbal.
6. **GELU float16 drift on ANE**: Test after CoreML conversion.
7. **Target widening width**: Test [0.3, 0.6, 1.0, 0.6, 0.3] vs single-frame.
8. **MERT layer selection**: Validate layer 5-6 vs layer 7-8 of MERT-95M.

---

## Appendix A: Dataset Summary

| Dataset | Size | Classes | Velocity | Audio Type | Role |
|---------|------|---------|----------|------------|------|
| E-GMD | 444h | GM (full) | MIDI GT | Electronic (TD-17) | Primary training |
| STAR Drums | 100.6h | 18 | Annotated | Re-synthesized + real accompaniment | Real-world training |
| ADTOF | 359h | 5 | No | Pre-computed spectrograms | Auxiliary/pre-training |
| MDB Drums | 23 tracks | 6+21 | No | Real acoustic | Evaluation only |
| ENST Drums | 3 drummers | ~8 | No | Real acoustic | Evaluation only |
| IDMT-SMT | 560 files | 3 | No | Real + separated | Evaluation only |
| MUSDB18-HQ | 150 tracks | N/A | N/A | Multi-stem music | Accompaniment mixing source |

### Local Paths

- E-GMD: `~/Documents/Datasets/e-gmd/` (129GB)
- STAR: `~/Documents/Datasets/star-drums/` (169GB)
- IDMT: `~/Documents/Datasets/idmt-smt-drums/` (622MB)
- MDB: `~/Documents/Datasets/mdb-drums/` (1.9GB)
- MUSDB18-HQ: `~/Documents/Datasets/musdb18hq/`

### External Resources

- ADTOF: [Zenodo 5624527](https://zenodo.org/records/5624527) / [GitHub](https://github.com/MZehren/ADTOF)
- RIRs: MIT IR Survey, BUT ReverbDB, EchoThief (~1,600 total)
- SFZ Kits: [sfzinstruments.github.io/drums/](https://sfzinstruments.github.io/drums/)

## Appendix B: Revision Log (Passes 2-5)

| # | Revision | Pass |
|---|---|---|
| 1 | (1,11) asymmetric temporal kernels | 2: Research Validity |
| 2 | MERT layer 5-6 of 95M | 2: Research Validity |
| 3 | Frozen BatchNorm protocol | 2: Research Validity |
| 4 | 14 base kits + augmentation → 42-56 virtual | 2: Research Validity |
| 5 | 3 separate fixed-shape CoreML models | 3: CoreML Feasibility |
| 6 | FiLM: pre-expand tensors, no broadcast | 3: CoreML Feasibility |
| 7 | Three-tier inference pipeline | 3: CoreML Feasibility |
| 8 | (B,C,1,T) 4D tensor format | 3: CoreML Feasibility |
| 9 | ml-ane-transformers attention pattern | 3: CoreML Feasibility |
| 10 | coremltools 8.2+ for sliced-Q | 3: CoreML Feasibility |
| 11 | BCE + target widening (ASL as ablation) | 4: Training Realism |
| 12 | ADTOF downgraded to auxiliary | 4: Training Realism |
| 13 | Revised 3-stage training pipeline | 4: Training Realism |
| 14 | STAR preprocessing steps | 4: Training Realism |
| 15 | Mix-training (50/50 isolated + accompaniment) | 5: Generalization |
| 16 | Standardized evaluation protocol | 5: Generalization |
| 17 | Realistic performance targets | 5: Generalization |
| 18 | U-Net decoder with skip connections | Post-plan: Architecture |
