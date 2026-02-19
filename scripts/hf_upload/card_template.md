---
license: {{LICENSE_SPDX}}
task_categories:
{{TASK_CATEGORIES}}
tags:
  - drums
  - percussion
  - music-information-retrieval
{{EXTRA_TAGS}}
pretty_name: "{{PRETTY_NAME}}"
size_categories:
  - {{SIZE_CATEGORY}}
---

# {{PRETTY_NAME}}

<!-- Optional: Embed an audio preview widget. HuggingFace dataset viewer
     renders audio columns automatically when the dataset has an Audio feature.
     For datasets without a loading script, you can embed a sample manually: -->
{{AUDIO_PREVIEW}}

## Quick Start

```python
from datasets import load_dataset

# Stream to avoid downloading the entire dataset
ds = load_dataset("schismaudio/{{REPO_NAME}}", streaming=True)

# Or download locally
ds = load_dataset("schismaudio/{{REPO_NAME}}")
```

## Dataset Description

{{DESCRIPTION}}

## Dataset Structure

### Data Fields

{{FIELDS_TABLE}}

<!-- Example fields table:
| Field | Type | Description |
|-------|------|-------------|
| `audio` | `Audio` | WAV audio file, 44.1kHz mono |
| `midi` | `string` | Path to aligned MIDI file |
| `drummer` | `string` | Anonymized drummer ID |
| `style` | `string` | Musical style (e.g., rock, jazz, latin) |
| `tempo` | `int` | BPM of the performance |
-->

### Data Splits

{{SPLITS_TABLE}}

<!-- Example splits table:
| Split | Examples | Duration | Description |
|-------|----------|----------|-------------|
| `train` | 8,068 | 320h | Training set |
| `validation` | 1,813 | 72h | Validation set (held-out drummers) |
| `test` | 1,235 | 52h | Test set (held-out drummers) |
-->

<!-- CLASS_TAXONOMY_START — Remove this section for datasets without
     onset/class annotations (e.g., RIR datasets, drum loop datasets). -->
## Class Taxonomy

{{CLASS_TAXONOMY}}

<!-- Example class taxonomy for ADT datasets:
The dataset uses the following drum instrument classes, mapped from
General MIDI percussion (channel 10):

| Class ID | Instrument | MIDI Note(s) | GM Name |
|----------|-----------|--------------|---------|
| 0 | Kick | 35, 36 | Acoustic Bass Drum, Bass Drum 1 |
| 1 | Snare | 38, 40 | Acoustic Snare, Electric Snare |
| 2 | Closed Hi-Hat | 42 | Closed Hi-Hat |
| 3 | Open Hi-Hat | 46 | Open Hi-Hat |
| 4 | Low Tom | 41, 43 | Low Floor Tom, High Floor Tom |
| 5 | Mid Tom | 45, 47 | Low Tom, Low-Mid Tom |
| 6 | High Tom | 48, 50 | Hi-Mid Tom, High Tom |
| 7 | Crash Cymbal | 49, 57 | Crash Cymbal 1, Crash Cymbal 2 |
| 8 | Ride Cymbal | 51, 59 | Ride Cymbal 1, Ride Cymbal 2 |
-->
<!-- CLASS_TAXONOMY_END -->

## Usage Examples

{{USAGE_EXAMPLES}}

<!-- Example usage code:
### Load and play audio

```python
from datasets import load_dataset
import sounddevice as sd

ds = load_dataset("schismaudio/{{REPO_NAME}}", split="train", streaming=True)
sample = next(iter(ds))
sd.play(sample["audio"]["array"], samplerate=sample["audio"]["sampling_rate"])
```

### Iterate with filtering

```python
ds = load_dataset("schismaudio/{{REPO_NAME}}", split="train")
rock_tracks = ds.filter(lambda x: x["style"] == "rock")
```
-->

## Dataset Creation

### Source Data

{{SOURCE_DATA}}

### Annotations

{{ANNOTATIONS}}

<!-- Describe:
     - Who created the annotations (human annotators, automatic alignment, etc.)
     - Annotation format (MIDI, CSV, TSV, TextGrid, etc.)
     - Quality control procedures (inter-annotator agreement, spot checks, etc.)
     - For synthesized datasets: rendering pipeline, software used, parameters
-->

## Known Limitations

{{LIMITATIONS}}

## Related Datasets

This dataset is part of the [Drum Audio Datasets](https://huggingface.co/collections/schismaudio/drum-audio-datasets) collection by [schismaudio](https://huggingface.co/schismaudio). Related datasets:

{{RELATED_DATASETS}}

<!-- Example related datasets:
- [schismaudio/e-gmd](https://huggingface.co/datasets/schismaudio/e-gmd) — 444 hours of electronic drum performances with aligned MIDI
- [schismaudio/groove-midi-dataset](https://huggingface.co/datasets/schismaudio/groove-midi-dataset) — 13.6 hours from the original Groove MIDI Dataset
- [schismaudio/stemgmd](https://huggingface.co/datasets/schismaudio/stemgmd) — 1,224 hours of multi-kit drum renders built on GMD
-->

## Citation

```bibtex
{{BIBTEX}}
```

## License

{{LICENSE_DESCRIPTION}}

<!-- Example license description:
This dataset is released under the
[Creative Commons Attribution 4.0 International License (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

You are free to share and adapt this dataset for any purpose, including
commercial use, as long as you give appropriate credit to the original authors.
-->
