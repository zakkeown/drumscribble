---
license: cc-by-4.0
task_categories:
  - audio-classification
tags:
  - drum-samples
  - percussion
  - one-shot
  - anechoic
  - audio-classification
pretty_name: "Dynamic Percussion Dataset"
size_categories:
  - 1K<n<10K
---

# Dynamic Percussion Dataset

## Quick Start

```python
from datasets import load_dataset

# Stream to avoid downloading the entire dataset
ds = load_dataset("schismaudio/dynamic-percussion", streaming=True)

# Or download locally
ds = load_dataset("schismaudio/dynamic-percussion")
```

## Dataset Description

The **Dynamic Percussion Dataset** is a collection of one-shot percussion samples recorded in an anechoic chamber at Tampere University. Audio is 44.1 kHz, 24-bit, mono WAV. Pre-calculated [OpenL3](https://github.com/marl/openl3) audio embeddings are included alongside the raw waveforms. The full dataset is approximately 174 MB.

Originally created as part of a Bachelor's thesis (Tampere University, 2020), the dataset was designed to support research in percussion sound classification and sample retrieval. The controlled anechoic recording environment provides clean, room-reflection-free samples.

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `audio` | `Audio` | WAV file at 44.1 kHz, 24-bit, mono |
| `filename` | `string` | Original filename |
| `instrument` | `string` | Percussion instrument label |
| `embedding` | `Sequence[float]` | Pre-calculated OpenL3 embedding vector |

### Data Splits

This dataset has no predefined splits. All samples are in the default `train` split.

## Usage Examples

### Load audio samples

```python
from datasets import load_dataset

ds = load_dataset("schismaudio/dynamic-percussion")

sample = ds["train"][0]
print(sample["filename"], sample["instrument"])
# Access raw audio array and sampling rate
audio_array = sample["audio"]["array"]
sr = sample["audio"]["sampling_rate"]  # 44100
```

### Use pre-calculated embeddings

```python
import numpy as np
from datasets import load_dataset

ds = load_dataset("schismaudio/dynamic-percussion")

embeddings = np.array([s["embedding"] for s in ds["train"]])
labels = [s["instrument"] for s in ds["train"]]
print(f"Embedding matrix: {embeddings.shape}")
```

## Dataset Creation

### Source Data

All samples were recorded in an anechoic chamber at Tampere University using various percussion instruments. The anechoic environment ensures each sample is free of room reflections and background noise, making the recordings suitable as clean one-shot sources for synthesis, augmentation, and classification research.

### Annotations

Instrument labels were assigned at recording time. OpenL3 embeddings were computed offline using the pre-trained music model.

## Known Limitations

- **Anechoic conditions:** The recording environment does not match real-world studio or live settings. Models trained on this data may not generalize to reverberant or noisy conditions.
- **Limited instrument diversity:** The dataset covers a subset of percussion instruments; rare or regional instruments are not represented.
- **Mono only:** No stereo recordings are included.

## Related Datasets

This dataset is part of the [Drum Audio Datasets](https://huggingface.co/collections/schismaudio/drum-audio-datasets) collection by [schismaudio](https://huggingface.co/schismaudio). Related datasets:

- [schismaudio/drum-percussion-kits](https://huggingface.co/datasets/schismaudio/drum-percussion-kits) — Broader collection of drum and percussion one-shots including SampleSwap and generated samples
- [schismaudio/vcsl-percussion](https://huggingface.co/datasets/schismaudio/vcsl-percussion) — Percussion samples from the Virtual Competitive Score Library

## Citation

```bibtex
@misc{dynamicpercussion2020,
  title     = {Dynamic Percussion Dataset},
  author    = {Tampere University},
  year      = {2020},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.3780109},
  url       = {https://zenodo.org/record/3780109}
}
```

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this dataset for any purpose, including commercial use, as long as you give appropriate credit to the original authors (Tampere University).
