# HuggingFace Drum Dataset Ecosystem — Design Doc

**Date:** 2026-02-19
**Author:** Zak Keown
**Status:** Approved

## Goal

Build a comprehensive, community-showcase-quality collection of drum and percussion datasets on HuggingFace under the `schismaudio` organization. This serves two purposes: (1) make drum transcription datasets accessible to the ADT research community, and (2) provide clean, reliable dataset hosting for the Drumscribble training pipeline.

## Scope

### Datasets to Host (16 total, ~1.5TB aggregate)

Only datasets with permissive licenses (CC-BY 4.0, CC0, Apache 2.0) are included. No CC-BY-NC or restrictive-license datasets.

#### Tier 1 — Core ADT Training Datasets (hcloud required)

| Dataset | Size (dl / extracted) | License | Source URL | HF Repo |
|---------|----------------------|---------|------------|---------|
| E-GMD | 90GB / 132GB | CC-BY 4.0 | magenta.withgoogle.com/datasets/e-gmd | `schismaudio/e-gmd` |
| StemGMD | 373GB / 1.13TB | CC-BY 4.0 | zenodo.org/records/7860223 | `schismaudio/stemgmd` |
| Slakh2100 | 105GB / ~500GB | CC-BY 4.0 | zenodo.org/records/4599666 | `schismaudio/slakh2100` |
| STAR Drums (partial) | TBD after license audit | Mixed CC (permissive subset) | zenodo.org/records/15690078 | `schismaudio/star-drums` |

#### Tier 2 — Smaller ADT & One-shot Datasets (local feasible)

| Dataset | Size | License | Source URL | HF Repo |
|---------|------|---------|------------|---------|
| GMD (Groove MIDI) | 4.76GB | CC-BY 4.0 | magenta.withgoogle.com/datasets/groove | `schismaudio/groove-midi-dataset` |
| Dynamic Percussion | 174MB | CC-BY 4.0 | zenodo.org/records/3780109 | `schismaudio/dynamic-percussion` |
| Drum & Perc Kits | 1.2GB | CC-BY 4.0 | zenodo.org/records/3994999 | `schismaudio/drum-percussion-kits` |
| VCSL (perc subset) | ~3-4GB | CC0 | github.com/sgossner/VCSL | `schismaudio/vcsl-percussion` |

#### Tier 3 — Drum Loop Datasets (local feasible)

| Dataset | Size | License | Source URL | HF Repo |
|---------|------|---------|------------|---------|
| Patchbanks Lo-Fi Drums | varies | CC-BY 4.0 | github.com/patchbanks/Lo-Fi-Drums-Dataset | `schismaudio/waivops-lofi-drums` |
| Patchbanks WRLD-LP | varies | CC-BY 4.0 | zenodo.org/records/8388266 | `schismaudio/waivops-world-percussion` |
| Patchbanks EDM-HSE | 7.6GB | CC-BY 4.0 | github.com/patchbanks/WaivOps-EDM-HSE | `schismaudio/waivops-edm-house` |
| Patchbanks RTRO-DRM | varies | CC-BY 4.0 | github.com/patchbanks/WaivOps-RTRO-DRM | `schismaudio/waivops-retro-drums` |
| Patchbanks HH-LFBB | varies | CC-BY 4.0 | github.com/patchbanks/WaivOps-HH-LFBB | `schismaudio/waivops-hiphop-lofi` |
| Patchbanks EDM-TR8 | 4.4GB | CC-BY 4.0 | github.com/patchbanks/WaivOps-EDM-TR8 | `schismaudio/waivops-edm-tr808` |

#### Tier 4 — Augmentation (RIRs)

| Dataset | Size | License | Source URL | HF Repo |
|---------|------|---------|------------|---------|
| OpenSLR RIRS_NOISES | 1.3GB | Apache 2.0 | openslr.org/28 | `schismaudio/rirs-noises` |
| dEchorate | small | CC-BY 4.0 | zenodo.org/records/4626590 | `schismaudio/dechorate` |

### Excluded Datasets (with reasons)

| Dataset | License | Reason |
|---------|---------|--------|
| ENST-Drums | Research-only (custom) | No redistribution allowed |
| RBMA 13 | All Rights Reserved | Copyrighted audio |
| IDMT-SMT-Drums | CC-BY-NC-ND 4.0 | NoDerivatives clause |
| MUSDB18 | CC-BY-NC-SA 4.0 | Restricted access, NC |
| MDB Drums | CC-BY-NC-SA 4.0 | NC; creators ask no republishing |
| ADTOF | CC-BY-NC-SA 4.0 | NC; spectrograms of copyrighted music |
| MusicNet | CC-BY 4.0 | No drums/percussion content |

## Repository Structure

Each dataset follows a consistent structure:

```
schismaudio/<dataset-name>/
├── README.md                    # Dataset card (showcase quality)
├── data/                        # Raw audio + annotations
│   ├── train/
│   ├── validation/
│   └── test/
├── features/                    # Pre-computed features (separate config)
│   ├── mel_spectrograms/        # 128-bin mel, 16kHz, hop=512
│   └── onset_labels/            # Frame-level binary, 26 classes
└── <dataset_name>.py            # HF datasets loading script (if needed)
```

For datasets without standard train/val/test splits, the original directory structure is preserved under `data/` and splits are documented in the card.

### HF Collection

A `schismaudio/drum-audio-datasets` collection aggregates all repos with:
- Overview of the drum dataset landscape
- Comparison table across datasets
- Recommendations for different use cases (transcription, separation, synthesis, augmentation)
- Cross-references between related datasets

### Existing Repos to Purge

- `zkeown/e-gmd-v1` — unreliable tar-based upload
- `zkeown/star-drums-v1` — unreliable tar-based upload

## Dataset Card Template

Every dataset card follows this standardized template for community-showcase quality:

```yaml
---
license: cc-by-4.0
task_categories:
  - audio-classification  # varies per dataset
tags:
  - drum-transcription
  - drums
  - percussion
  - music-information-retrieval
pretty_name: "Full Dataset Name"
size_categories:
  - 10K<n<100K  # varies
dataset_info:
  - config_name: raw
    ...
  - config_name: features
    ...
---
```

### Card Sections

1. **Title + Audio Preview** — Embedded player for sample tracks (HF `Audio` column support)
2. **Quick Start** — `load_dataset()` code snippet
3. **Dataset Description** — 2-3 paragraphs: what, who, why
4. **Dataset Structure** — Data instances, fields table, splits table
5. **Class Taxonomy** — Full drum class mapping with MIDI notes, grouped by instrument family
6. **Visualizations** — Mel spectrogram samples, onset timelines, class distribution histograms (generated images embedded)
7. **Usage Examples** — Code for loading, visualization, training loop integration
8. **Dataset Creation** — Source data, recording setup, annotation process, quality control
9. **Known Limitations** — Biases, gaps, quality issues
10. **Related Datasets** — Cross-references to other `schismaudio/` datasets
11. **Citation** — BibTeX entry from original paper
12. **License** — Details with link to full text

### Showcase Elements

- Embedded audio player widgets using HF dataset viewer
- Pre-generated visualization images (spectrograms, distributions)
- Consistent GM drum taxonomy documentation across all ADT datasets
- Code examples that work with `datasets` library
- Cross-dataset comparison in collection page

## Infrastructure

### Phase 1 — Purge & Setup

1. Delete `zkeown/e-gmd-v1` and `zkeown/star-drums-v1`
2. Create `schismaudio` org on HF (if not exists)
3. Create empty dataset repos for all 16 datasets
4. Create `schismaudio/drum-audio-datasets` collection

### Phase 2 — Local Uploads (Tier 2-4, ~30GB total)

Upload from local machine (these all fit comfortably within 1TB):
- GMD, Dynamic Percussion, Drum & Perc Kits, VCSL percussion subset
- All 6 Patchbanks WaivOps datasets
- OpenSLR RIRS, dEchorate

Workflow per dataset:
1. Download from source (Zenodo/GitHub/GCS)
2. Extract, verify integrity
3. Restructure into HF format (data/ directory, splits)
4. Generate pre-computed features (mel spectrograms, onset labels where applicable)
5. Write dataset card from template
6. `huggingface-cli upload schismaudio/<name> ./data`
7. Verify `load_dataset()` works
8. Clean up local files before next dataset

### Phase 3 — hcloud for Tier 1 Datasets

Spin up Hetzner Cloud servers for large datasets:

| Dataset | Server Type | NVMe | Est. Cost | Est. Time |
|---------|------------|------|-----------|-----------|
| E-GMD | CX32 | 160GB | ~$2-3 | 2-4 hours |
| Slakh2100 | CX52 | 480GB | ~$5-8 | 4-8 hours |
| STAR (partial) | CX32-CX52 | 160-480GB | ~$2-5 | 2-4 hours |
| StemGMD | CCX53 + 1TB volume | 960GB + volume | ~$15-25 | 8-16 hours |

hcloud workflow per dataset:
1. `hcloud server create --type <type> --image ubuntu-24.04 --name dataset-upload`
2. SSH in, install `hf` CLI + `pip install huggingface_hub`
3. Authenticate with HF token
4. Download dataset from source
5. Extract, restructure, generate features
6. Upload to HF Hub (use `upload_large_folder` for TB-scale)
7. Verify upload integrity
8. `hcloud server delete dataset-upload`

#### StemGMD Special Handling

At 1.13TB extracted, StemGMD requires careful handling:
- Option A: Large server with attached volume (960GB NVMe + 1TB volume = ~2TB working space)
- Option B: Stream-process — download one Zenodo archive part, extract, upload, delete, repeat
- Option B is cheaper but slower and risks partial upload state

#### STAR Drums License Audit

Before uploading STAR, audit per-track licenses:
1. Download the full STAR dataset (or just the LICENSE folder)
2. Parse per-track license assignments
3. Identify tracks with CC-BY or more permissive licenses
4. Exclude tracks with NC/ND/research-only licenses (especially MUSDB18-derived tracks)
5. Document which tracks are included and why in the dataset card

### Phase 4 — Feature Pre-computation

For each ADT dataset (E-GMD, GMD, StemGMD, Slakh2100, STAR partial), generate:
- **Mel spectrograms**: 128 bins, 16kHz sample rate, hop_length=512 (31.25 fps)
- **Onset labels**: Frame-level binary arrays, 26 GM drum classes, aligned to spectrogram timing
- **Storage format**: Parquet files for HF datasets integration
- **Config**: Separate `features` config in dataset YAML so users can `load_dataset("schismaudio/e-gmd", "features")`

This can be done on hcloud during the upload phase (same server session) or as a separate pass.

### Phase 5 — Polish & Cross-reference

1. Generate visualization images for all dataset cards (spectrograms, distributions, class breakdowns)
2. Finalize HF collection page with overview and comparison table
3. Cross-reference related datasets in each card
4. Test `load_dataset()` for all configs (raw + features)
5. Announce on relevant channels (ISMIR community, Twitter/X, etc.)

## STAR Drums License Audit Detail

The STAR dataset derives from MUSDB18 sources with mixed licensing. The audit process:

1. Download `LICENSE/` folder from Zenodo (or full dataset if needed)
2. For each track, check the source license:
   - **DSD100 tracks**: Mostly CC-BY-NC or CC-BY-NC-SA → exclude
   - **MedleyDB tracks**: CC-BY-NC-SA → exclude
   - **Other sources**: Check individually
3. The "pre-mixed" downloadable portion vs. the "requires local mixing script" portion may have different license implications
4. Document the filtering criteria and exact track list in the dataset card
5. If the redistributable subset is too small (<10% of tracks), consider skipping STAR entirely

## Success Criteria

- All 16 datasets uploaded with working `load_dataset()` calls
- Every dataset card follows the template with all 12 sections
- Audio preview widgets work in HF dataset viewer
- Pre-computed features available as separate config for ADT datasets
- Collection page provides clear overview and comparison
- Total cost for hcloud stays under ~$50
- Existing unreliable repos purged

## Estimated Timeline

| Phase | Duration | Datasets |
|-------|----------|----------|
| Phase 1: Setup | 1 hour | N/A |
| Phase 2: Local uploads | 2-3 days | 12 datasets |
| Phase 3: hcloud uploads | 2-3 days | 4 datasets |
| Phase 4: Features | 1-2 days | 5 ADT datasets |
| Phase 5: Polish | 1 day | All |
| **Total** | **~7-10 days** | **16 datasets** |

## References

- E-GMD: Callender et al. (2020). arXiv:2004.00188
- GMD: Gillick et al. (2019). ICML 2019
- StemGMD: Ferroni et al. (2023). Proc. ISMIR 2023
- Slakh2100: Manilow et al. (2019). Proc. ISMIR 2019
- STAR Drums: Weber et al. (2025). TISMIR 8(1)
- Dynamic Percussion Dataset: Bachelor's thesis, Tampere University (2020)
- Patchbanks WaivOps: CC-BY 4.0, various GitHub repos
- OpenSLR RIRS_NOISES: Apache 2.0, openslr.org/28
- dEchorate: CC-BY 4.0, zenodo.org/records/4626590
