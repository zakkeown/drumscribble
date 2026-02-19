---
license: cc-by-4.0
task_categories:
  - audio-classification
tags:
  - drum-samples
  - one-shot
  - sample-library
  - audio-classification
pretty_name: "Drum & Percussion Kits"
size_categories:
  - 1K<n<10K
---

# Drum & Percussion Kits

## Quick Start

```python
from datasets import load_dataset

# Stream to avoid downloading the entire dataset
ds = load_dataset("schismaudio/drum-percussion-kits", streaming=True)

# Or download locally
ds = load_dataset("schismaudio/drum-percussion-kits")
```

## Dataset Description

**Drum & Percussion Kits** is a diverse collection of drum and percussion one-shot samples assembled for use in synthesis and data augmentation pipelines. It combines free samples from [SampleSwap](https://sampleswap.org/), organic one-shot recordings, and 850+ programmatically generated samples. The full dataset is approximately 1.2 GB, distributed across 4 tar archives.

The dataset is intended as a broad-coverage sample library for training drum transcription and synthesis models, providing timbral variety beyond what any single recorded kit can offer.

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio` | `Audio` | WAV one-shot sample |
| `filename` | `string` | Original filename |
| `instrument` | `string` | Drum or percussion instrument label |
| `source` | `string` | Provenance tag: `sampleswap`, `organic`, or `generated` |

### Data Splits

This dataset has no predefined splits. All samples are in the default `train` split.

### Archives

The dataset is packaged as 4 tar archives. Each archive contains a subset of the sample library organized by instrument family.

## Usage Examples

### Load and inspect samples

```python
from datasets import load_dataset

ds = load_dataset("schismaudio/drum-percussion-kits")

sample = ds["train"][0]
print(sample["filename"], sample["instrument"], sample["source"])
audio_array = sample["audio"]["array"]
sr = sample["audio"]["sampling_rate"]
```

### Filter by provenance

```python
from datasets import load_dataset

ds = load_dataset("schismaudio/drum-percussion-kits")

organic = ds["train"].filter(lambda x: x["source"] == "organic")
print(f"Organic samples: {len(organic)}")
```

## Dataset Creation

### Source Data

The collection draws from three sources:

- **SampleSwap:** Free-to-use drum samples from the SampleSwap community library, covering a wide range of drum machine and acoustic kit sounds.
- **Organic one-shots:** Individually recorded percussion hits captured in various acoustic environments.
- **Generated samples:** 850+ samples synthesized programmatically to expand coverage of underrepresented timbres and velocities.

### Annotations

Instrument labels and source tags were assigned during curation. No onset or velocity annotations are included — this is a one-shot sample library, not a transcription dataset.

## Known Limitations

- **Mixed provenance:** Samples originate from multiple sources with different recording conditions, bit depths, and sample rates. Normalization and quality may be inconsistent.
- **Generated samples:** The 850+ generated samples may not exhibit the same acoustic realism as recorded samples. Models trained on this data should be evaluated on real-world audio.
- **No standardized splits:** Without defined train/test splits, users should partition the data carefully to avoid data leakage.

## Related Datasets

This dataset is part of the [Drum Audio Datasets](https://huggingface.co/collections/schismaudio/drum-audio-datasets) collection by [schismaudio](https://huggingface.co/schismaudio). Related datasets:

- [schismaudio/dynamic-percussion](https://huggingface.co/datasets/schismaudio/dynamic-percussion) — Anechoic one-shot percussion samples with OpenL3 embeddings (Tampere University, 2020)
- [schismaudio/vcsl-percussion](https://huggingface.co/datasets/schismaudio/vcsl-percussion) — Percussion samples from the Virtual Competitive Score Library

## Citation

```bibtex
@misc{drumpercussionkits,
  title     = {Drum \& Percussion Kits},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.3994999},
  url       = {https://zenodo.org/record/3994999}
}
```

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this dataset for any purpose, including commercial use, as long as you give appropriate credit to the original authors.
