# HuggingFace Drum Dataset Ecosystem — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upload 16 drum/percussion datasets to HuggingFace under `schismaudio` org with community-showcase-quality dataset cards, pre-computed features, and a unified collection page.

**Architecture:** Each dataset gets its own HF repo with raw audio + annotations, pre-computed features as a separate config, and a standardized dataset card. Small datasets (<10GB) upload from local machine; large datasets (>10GB) use Hetzner Cloud ephemeral servers. A reusable Python upload script handles the common workflow.

**Tech Stack:** `huggingface_hub` Python library, `hf` CLI, `hcloud` CLI, `librosa` for feature extraction, `datasets` library for verification.

**Design doc:** `docs/plans/2026-02-19-hf-drum-dataset-ecosystem-design.md`

---

### Task 1: Purge Old Repos & Create Dataset Repos

**Files:**
- None (all HF Hub operations)

**Step 1: Check for and delete old repos**

```bash
# Check if old repos exist
hf repo ls zkeown --type dataset 2>/dev/null || echo "No datasets under zkeown"

# Delete if they exist (will prompt for confirmation)
hf repo delete zkeown/e-gmd-v1 --type dataset --yes 2>/dev/null || echo "e-gmd-v1 not found"
hf repo delete zkeown/star-drums-v1 --type dataset --yes 2>/dev/null || echo "star-drums-v1 not found"
```

**Step 2: Create all 16 dataset repos under schismaudio**

```bash
for repo in e-gmd groove-midi-dataset stemgmd slakh2100 star-drums \
            dynamic-percussion drum-percussion-kits vcsl-percussion \
            waivops-lofi-drums waivops-world-percussion waivops-edm-house \
            waivops-retro-drums waivops-hiphop-lofi waivops-edm-tr808 \
            rirs-noises dechorate; do
    echo "Creating schismaudio/$repo..."
    hf repo create "schismaudio/$repo" --type dataset || echo "Already exists: $repo"
done
```

**Step 3: Create the HF collection**

Go to https://huggingface.co/collections and create `schismaudio/drum-audio-datasets` manually (HF CLI doesn't support collection creation). Add a placeholder description:

> A comprehensive collection of drum and percussion datasets for automatic drum transcription (ADT), drum source separation, synthesis, and augmentation. All datasets use permissive licenses (CC-BY 4.0, CC0, Apache 2.0).

**Step 4: Verify**

```bash
hf repo ls schismaudio --type dataset
```

Expected: 16 dataset repos listed.

**Step 5: Commit**

No code changes — this is all HF Hub ops.

---

### Task 2: Create Dataset Card Template

**Files:**
- Create: `scripts/hf_upload/card_template.md`

**Step 1: Write the template**

Create `scripts/hf_upload/card_template.md` with the standardized dataset card structure from the design doc. Use `{{PLACEHOLDER}}` syntax for fields that vary per dataset.

```markdown
---
license: {{LICENSE_SPDX}}
task_categories:
  - {{TASK_CATEGORY}}
tags:
  - drums
  - percussion
  - {{EXTRA_TAGS}}
pretty_name: "{{PRETTY_NAME}}"
size_categories:
  - {{SIZE_CATEGORY}}
---

# {{PRETTY_NAME}}

{{AUDIO_PREVIEW}}

## Quick Start

\```python
from datasets import load_dataset

ds = load_dataset("schismaudio/{{REPO_NAME}}")
\```

## Dataset Description

{{DESCRIPTION}}

## Dataset Structure

### Data Fields

{{FIELDS_TABLE}}

### Data Splits

{{SPLITS_TABLE}}

## Class Taxonomy

{{CLASS_TAXONOMY}}

## Usage Examples

{{USAGE_EXAMPLES}}

## Dataset Creation

### Source Data

{{SOURCE_DATA}}

### Annotations

{{ANNOTATIONS}}

## Known Limitations

{{LIMITATIONS}}

## Related Datasets

This dataset is part of the [Drum Audio Datasets](https://huggingface.co/collections/schismaudio/drum-audio-datasets) collection. Related datasets:

{{RELATED_DATASETS}}

## Citation

\```bibtex
{{BIBTEX}}
\```

## License

{{LICENSE_DESCRIPTION}}
```

**Step 2: Commit**

```bash
git add scripts/hf_upload/card_template.md
git commit -m "feat: add HF dataset card template"
```

---

### Task 3: Create Upload Tooling Script

**Files:**
- Create: `scripts/hf_upload/upload_dataset.py`
- Create: `scripts/hf_upload/__init__.py` (empty)

**Step 1: Write the upload script**

Create `scripts/hf_upload/upload_dataset.py` — a reusable CLI tool that handles the common workflow for any dataset:

```python
#!/usr/bin/env python3
"""Upload a local dataset directory to HuggingFace Hub.

Usage:
    python scripts/hf_upload/upload_dataset.py \
        --repo schismaudio/e-gmd \
        --local-dir /tmp/datasets/e-gmd \
        --readme /tmp/datasets/e-gmd/README.md

Handles:
    - Uploading files/folders to HF Hub
    - Large folder uploads with resume support
    - Verification via load_dataset()
"""
import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_large_folder


def upload_dataset(repo_id: str, local_dir: str, readme_path: str | None = None):
    api = HfApi()

    local = Path(local_dir)
    if not local.exists():
        print(f"Error: {local} does not exist")
        sys.exit(1)

    # Upload README first if provided
    if readme_path:
        readme = Path(readme_path)
        if readme.exists():
            print(f"Uploading README.md to {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(readme),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )

    # Calculate total size
    total_bytes = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
    total_gb = total_bytes / (1024**3)
    print(f"Total upload size: {total_gb:.2f} GB")

    if total_gb > 50:
        print("Using upload_large_folder for large dataset...")
        upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local),
        )
    else:
        print("Using upload_folder...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local),
        )

    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")


def verify_dataset(repo_id: str):
    """Try loading the dataset to verify it works."""
    try:
        from datasets import load_dataset
        print(f"Verifying {repo_id} with load_dataset()...")
        ds = load_dataset(repo_id, split="train", streaming=True)
        sample = next(iter(ds))
        print(f"Verification passed. Sample keys: {list(sample.keys())}")
        return True
    except Exception as e:
        print(f"Verification failed: {e}")
        print("This may be expected if the dataset needs a custom loading script.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument("--repo", required=True, help="HF repo ID (e.g. schismaudio/e-gmd)")
    parser.add_argument("--local-dir", required=True, help="Local directory to upload")
    parser.add_argument("--readme", help="Path to README.md dataset card")
    parser.add_argument("--verify", action="store_true", help="Verify with load_dataset() after upload")
    args = parser.parse_args()

    upload_dataset(args.repo, args.local_dir, args.readme)
    if args.verify:
        verify_dataset(args.repo)
```

**Step 2: Create empty __init__.py**

```bash
touch scripts/hf_upload/__init__.py
```

**Step 3: Verify script syntax**

```bash
python -c "import ast; ast.parse(open('scripts/hf_upload/upload_dataset.py').read()); print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add scripts/hf_upload/
git commit -m "feat: add reusable HF dataset upload script"
```

---

### Task 4: Upload GMD (Groove MIDI Dataset)

First local upload — 4.76GB, good test of the workflow.

**Files:**
- Create: `scripts/hf_upload/cards/groove-midi-dataset.md`

**Step 1: Download GMD**

```bash
mkdir -p /tmp/datasets/gmd
cd /tmp/datasets/gmd

# Download from Google (check exact URL on magenta.withgoogle.com/datasets/groove)
# MIDI-only is 3.11 MB, full audio+MIDI is 4.76 GB
curl -L -o groove-v1.0.0.zip "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip"
unzip groove-v1.0.0.zip
```

**Step 2: Inspect structure and write dataset card**

Inspect the extracted directory structure, splits, and metadata CSV. Then write `scripts/hf_upload/cards/groove-midi-dataset.md` using the template from Task 2, filling in:

- **License:** CC-BY 4.0
- **Citation:** Gillick et al. (2019). "Learning to Groove with Inverse Sequence Transformations." ICML 2019
- **Description:** 13.6 hours of aligned MIDI and audio from 10 drummers on a Roland TD-11. 1,150 MIDI files, over 22,000 measures. 9 canonical drum instrument classes mapped from 22 MIDI pitches.
- **Task categories:** `audio-classification`
- **Tags:** `drum-transcription`, `midi`, `groove`, `rhythm`
- **Classes:** 9 canonical: kick, snare, closed hi-hat, open hi-hat, low tom, mid tom, high tom, crash, ride
- **Splits:** train/validation/test (documented in metadata CSV)

**Step 3: Upload to HF**

```bash
python scripts/hf_upload/upload_dataset.py \
    --repo schismaudio/groove-midi-dataset \
    --local-dir /tmp/datasets/gmd/groove \
    --readme scripts/hf_upload/cards/groove-midi-dataset.md
```

**Step 4: Verify on HF**

Visit https://huggingface.co/datasets/schismaudio/groove-midi-dataset and confirm:
- Files are visible in the file browser
- README renders correctly with audio preview
- Dataset viewer works (if HF auto-detects the format)

**Step 5: Clean up and commit**

```bash
rm -rf /tmp/datasets/gmd
git add scripts/hf_upload/cards/groove-midi-dataset.md
git commit -m "feat: upload GMD to schismaudio/groove-midi-dataset"
```

---

### Task 5: Upload Dynamic Percussion Dataset

**Files:**
- Create: `scripts/hf_upload/cards/dynamic-percussion.md`

**Step 1: Download from Zenodo**

```bash
mkdir -p /tmp/datasets/dynamic-perc
cd /tmp/datasets/dynamic-perc
# Zenodo record 3780109
curl -L -o dynamic-percussion.zip "https://zenodo.org/records/3780109/files/DynamicPercussionDataset.zip?download=1"
unzip dynamic-percussion.zip
```

**Step 2: Inspect and write dataset card**

- **License:** CC-BY 4.0
- **Description:** One-shot percussion samples recorded in an anechoic chamber. 44.1kHz, 24-bit, mono WAV. Pre-calculated OpenL3 embeddings included.
- **Tags:** `drum-samples`, `percussion`, `one-shot`, `anechoic`
- **Task:** `audio-classification`

**Step 3: Upload**

```bash
python scripts/hf_upload/upload_dataset.py \
    --repo schismaudio/dynamic-percussion \
    --local-dir /tmp/datasets/dynamic-perc \
    --readme scripts/hf_upload/cards/dynamic-percussion.md
```

**Step 4: Verify and clean up**

```bash
rm -rf /tmp/datasets/dynamic-perc
git add scripts/hf_upload/cards/dynamic-percussion.md
git commit -m "feat: upload Dynamic Percussion to schismaudio/dynamic-percussion"
```

---

### Task 6: Upload Drum & Percussion Kits

**Files:**
- Create: `scripts/hf_upload/cards/drum-percussion-kits.md`

**Step 1: Download from Zenodo**

```bash
mkdir -p /tmp/datasets/drum-kits
cd /tmp/datasets/drum-kits
# Zenodo record 3994999 — 4 tar files
for f in drum_kits_1.tar drum_kits_2.tar drum_kits_3.tar drum_kits_4.tar; do
    curl -L -o "$f" "https://zenodo.org/records/3994999/files/${f}?download=1"
    tar xf "$f"
    rm "$f"
done
```

**Step 2: Write dataset card**

- **License:** CC-BY 4.0
- **Description:** Free drum samples from SampleSwap + organic one-shots + 850+ generated samples. 1.2GB total.
- **Tags:** `drum-samples`, `one-shot`, `sample-library`

**Step 3: Upload, verify, clean up, commit**

Same pattern as Task 5. Upload to `schismaudio/drum-percussion-kits`.

---

### Task 7: Upload VCSL Percussion Subset

**Files:**
- Create: `scripts/hf_upload/cards/vcsl-percussion.md`

**Step 1: Clone VCSL and extract percussion**

```bash
mkdir -p /tmp/datasets/vcsl
cd /tmp/datasets/vcsl
git clone --depth 1 https://github.com/sgossner/VCSL.git
# Extract only percussion-related directories
# Inspect the directory structure first to find percussion content
ls VCSL/
```

**Step 2: Filter to percussion content only**

After inspecting the structure, copy only percussion/drum directories to an upload folder. Document what was included and what was filtered out.

**Step 3: Write dataset card**

- **License:** CC0 (Public Domain)
- **Description:** Percussion subset of the Versilian Community Sample Library. CC0 — absolutely no restrictions.
- **Tags:** `drum-samples`, `percussion`, `cc0`, `public-domain`, `sample-library`

**Step 4: Upload, verify, clean up, commit**

Upload to `schismaudio/vcsl-percussion`.

---

### Task 8: Upload Patchbanks WaivOps Datasets (batch of 6)

These 6 datasets share the same structure (WAV loops + JSON labels, CC-BY 4.0). Process them as a batch.

**Files:**
- Create: `scripts/hf_upload/cards/waivops-lofi-drums.md`
- Create: `scripts/hf_upload/cards/waivops-world-percussion.md`
- Create: `scripts/hf_upload/cards/waivops-edm-house.md`
- Create: `scripts/hf_upload/cards/waivops-retro-drums.md`
- Create: `scripts/hf_upload/cards/waivops-hiphop-lofi.md`
- Create: `scripts/hf_upload/cards/waivops-edm-tr808.md`

**Step 1: Download all 6 datasets**

```bash
mkdir -p /tmp/datasets/waivops
cd /tmp/datasets/waivops

# GitHub repos (clone)
git clone --depth 1 https://github.com/patchbanks/Lo-Fi-Drums-Dataset.git lofi-drums
git clone --depth 1 https://github.com/patchbanks/WaivOps-EDM-HSE.git edm-house
git clone --depth 1 https://github.com/patchbanks/WaivOps-RTRO-DRM.git retro-drums
git clone --depth 1 https://github.com/patchbanks/WaivOps-HH-LFBB.git hiphop-lofi
git clone --depth 1 https://github.com/patchbanks/WaivOps-EDM-TR8.git edm-tr808

# Zenodo (WRLD-LP)
curl -L -o wrld-lp.zip "https://zenodo.org/records/8388266/files/WaivOps-WRLD-LP.zip?download=1"
unzip wrld-lp.zip -d world-percussion
```

**Step 2: Inspect structure of each**

For each dataset, check:
- Number of WAV files, total size
- Label format (JSON, CSV, paired filenames?)
- Any README or license files to include

**Step 3: Write 6 dataset cards**

All follow the same template with per-dataset specifics:

| Repo | Tracks | Genre | Tempo Range |
|------|--------|-------|-------------|
| `waivops-lofi-drums` | 10,000 | Lo-fi | Various |
| `waivops-world-percussion` | 3,162 | World percussion | Various |
| `waivops-edm-house` | 8,000 | House music | 120-130 BPM |
| `waivops-retro-drums` | 2,138 | 1980s electronic | Various |
| `waivops-hiphop-lofi` | 3,332 | Lo-fi hip-hop | 60-90 BPM |
| `waivops-edm-tr808` | 3,790 | TR-808/electro | Various |

**Step 4: Upload each dataset**

```bash
for dataset in lofi-drums world-percussion edm-house retro-drums hiphop-lofi edm-tr808; do
    repo_map=("lofi-drums:waivops-lofi-drums" "world-percussion:waivops-world-percussion" \
              "edm-house:waivops-edm-house" "retro-drums:waivops-retro-drums" \
              "hiphop-lofi:waivops-hiphop-lofi" "edm-tr808:waivops-edm-tr808")
    # Upload each one
    python scripts/hf_upload/upload_dataset.py \
        --repo "schismaudio/waivops-${dataset}" \
        --local-dir "/tmp/datasets/waivops/${dataset}" \
        --readme "scripts/hf_upload/cards/waivops-${dataset}.md"
done
```

**Step 5: Verify all 6 repos, clean up, commit**

```bash
rm -rf /tmp/datasets/waivops
git add scripts/hf_upload/cards/waivops-*.md
git commit -m "feat: upload 6 Patchbanks WaivOps drum loop datasets"
```

---

### Task 9: Upload RIR Datasets (OpenSLR + dEchorate)

**Files:**
- Create: `scripts/hf_upload/cards/rirs-noises.md`
- Create: `scripts/hf_upload/cards/dechorate.md`

**Step 1: Download OpenSLR RIRS_NOISES**

```bash
mkdir -p /tmp/datasets/rirs
cd /tmp/datasets/rirs
curl -L -o rirs_noises.zip "https://www.openslr.org/resources/28/rirs_noises.zip"
unzip rirs_noises.zip
```

**Step 2: Download dEchorate**

```bash
mkdir -p /tmp/datasets/dechorate
cd /tmp/datasets/dechorate
# Zenodo record 4626590
curl -L -o dechorate.zip "https://zenodo.org/records/4626590/files/dEchorate.zip?download=1"
unzip dechorate.zip
```

**Step 3: Write dataset cards**

OpenSLR RIRS:
- **License:** Apache 2.0
- **Description:** Simulated and real RIRs + isotropic/point-source noises. Includes RIRs from RWCP, REVERB challenge, Aachen AIR databases, plus MUSAN noise. 16kHz, 16-bit.
- **Tags:** `room-impulse-response`, `audio-augmentation`, `rir`, `noise`

dEchorate:
- **License:** CC-BY 4.0
- **Description:** Calibrated multichannel RIRs with echo timing annotations and 3D positions. Cuboid room with configurable walls.
- **Tags:** `room-impulse-response`, `audio-augmentation`, `rir`, `multichannel`

**Step 4: Upload both, verify, clean up, commit**

---

### Task 10: STAR Drums License Audit

Before uploading STAR, audit which tracks have permissive licenses.

**Files:**
- Create: `scripts/hf_upload/star_license_audit.py`
- Create: `scripts/hf_upload/star_license_audit_results.json`

**Step 1: Examine STAR license structure**

The STAR dataset (local at `~/Documents/Datasets/star-drums/`) has a LICENSE folder or per-track license metadata. Inspect it:

```bash
ls ~/Documents/Datasets/star-drums/
# Look for LICENSE folder, README, or metadata files
find ~/Documents/Datasets/star-drums/ -name "LICENSE*" -o -name "license*" -o -name "*.csv" | head -20
```

**Step 2: Write license audit script**

Create `scripts/hf_upload/star_license_audit.py` that:
1. Reads the per-track license assignments from STAR's metadata
2. Classifies each track as permissive (CC-BY, CC0) or restrictive (CC-BY-NC, CC-BY-NC-SA, etc.)
3. Outputs a JSON file listing which tracks can be redistributed
4. Prints summary statistics (how many tracks total, how many permissive, how many excluded)

```python
#!/usr/bin/env python3
"""Audit STAR Drums per-track licenses to determine redistributable subset."""
import json
from pathlib import Path

STAR_ROOT = Path.home() / "Documents" / "Datasets" / "star-drums"
PERMISSIVE_LICENSES = {"CC-BY-4.0", "CC-BY-3.0", "CC-BY-2.0", "CC0-1.0", "CC0"}

def audit():
    # TODO: Parse actual STAR license metadata format (inspect in Step 1)
    # This is a skeleton — adapt after seeing the actual structure
    pass

if __name__ == "__main__":
    audit()
```

**Step 3: Run the audit**

```bash
python scripts/hf_upload/star_license_audit.py
```

Review the output. If the redistributable subset is <10% of tracks, consider skipping STAR entirely per the design doc.

**Step 4: Document findings**

Save results to `scripts/hf_upload/star_license_audit_results.json` and update the design doc with findings.

**Step 5: Commit**

```bash
git add scripts/hf_upload/star_license_audit.py scripts/hf_upload/star_license_audit_results.json
git commit -m "feat: STAR Drums license audit — identify redistributable tracks"
```

---

### Task 11: Create hcloud Bootstrap Script

Reusable script for spinning up Hetzner Cloud servers for large dataset uploads.

**Files:**
- Create: `scripts/hf_upload/hcloud_bootstrap.sh`

**Step 1: Write the bootstrap script**

```bash
#!/usr/bin/env bash
# Bootstrap a Hetzner Cloud server for HF dataset uploads.
# Usage: ./hcloud_bootstrap.sh <server-type> <server-name>
# Example: ./hcloud_bootstrap.sh cx32 egmd-upload
set -euo pipefail

SERVER_TYPE="${1:?Usage: $0 <server-type> <server-name>}"
SERVER_NAME="${2:?Usage: $0 <server-type> <server-name>}"

echo "=== Creating server $SERVER_NAME (type: $SERVER_TYPE) ==="
hcloud server create \
    --type "$SERVER_TYPE" \
    --image ubuntu-24.04 \
    --name "$SERVER_NAME" \
    --ssh-key default

IP=$(hcloud server ip "$SERVER_NAME")
echo "Server IP: $IP"
echo "Waiting for SSH..."
sleep 10

# Install dependencies
ssh -o StrictHostKeyChecking=no "root@$IP" bash <<'REMOTE'
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv unzip aria2
python3 -m venv /opt/hf-env
source /opt/hf-env/bin/activate
pip install -q huggingface_hub datasets
# Install hf CLI
pip install -q huggingface_hub[cli]
echo "Done. Run: source /opt/hf-env/bin/activate"
echo "Then: huggingface-cli login --token <YOUR_TOKEN>"
REMOTE

echo ""
echo "=== Server ready ==="
echo "SSH: ssh root@$IP"
echo "Activate env: source /opt/hf-env/bin/activate"
echo "Login to HF: huggingface-cli login --token \$HF_TOKEN"
echo ""
echo "When done: hcloud server delete $SERVER_NAME"
```

**Step 2: Make executable and commit**

```bash
chmod +x scripts/hf_upload/hcloud_bootstrap.sh
git add scripts/hf_upload/hcloud_bootstrap.sh
git commit -m "feat: add hcloud bootstrap script for dataset uploads"
```

---

### Task 12: Upload E-GMD via hcloud

**Files:**
- Create: `scripts/hf_upload/cards/e-gmd.md`
- Create: `scripts/hf_upload/hcloud_egmd.sh`

**Step 1: Write E-GMD dataset card**

This is the crown jewel — 444 hours, CC-BY 4.0, full MIDI + velocity + audio. Write a showcase-quality card:

- **License:** CC-BY 4.0
- **Citation:** Callender, L., Hawthorne, C., & Engel, J. (2020). "Improving Perceptual Quality of Drum Transcription with the Expanded Groove MIDI Dataset." arXiv:2004.00188
- **Description:** 444 hours of audio from 43 drum kits on a Roland TD-17. 45,537 sequences from 1,059 unique performances by 10 drummers. Aligned MIDI with velocity, CSV metadata (drummer ID, session, style, BPM, beat type, time signature, duration, kit name).
- **Tags:** `drum-transcription`, `midi`, `velocity`, `electronic-drums`, `groove`
- **Classes:** Raw MIDI pitches (Roland TD-17 mapping) — document the full pitch-to-instrument mapping
- **Splits:** By drummer (document which drummers in which split)
- **Size category:** `100K<n<1M` (45,537 sequences)

**Step 2: Write hcloud upload script**

Create `scripts/hf_upload/hcloud_egmd.sh` — a script to run ON the hcloud server:

```bash
#!/usr/bin/env bash
# Run this ON the hcloud server after bootstrap.
# Expects: HF_TOKEN env var set, /opt/hf-env activated.
set -euo pipefail

source /opt/hf-env/bin/activate

echo "=== Downloading E-GMD ==="
cd /root
mkdir -p egmd-work
cd egmd-work

# Download from Google Cloud Storage
# Check exact URL — may need to use gsutil or direct HTTP
curl -L -o e-gmd-v1.0.0.zip \
    "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip"

echo "=== Extracting ==="
unzip e-gmd-v1.0.0.zip
rm e-gmd-v1.0.0.zip

echo "=== Uploading to HF ==="
huggingface-cli upload schismaudio/e-gmd ./e-gmd-v1.0.0/ --repo-type dataset

echo "=== Done ==="
echo "Verify at: https://huggingface.co/datasets/schismaudio/e-gmd"
```

**Step 3: Spin up server and run**

```bash
# From local machine
./scripts/hf_upload/hcloud_bootstrap.sh cx32 egmd-upload

# SSH in
IP=$(hcloud server ip egmd-upload)
scp scripts/hf_upload/hcloud_egmd.sh "root@$IP:/root/"
scp scripts/hf_upload/cards/e-gmd.md "root@$IP:/root/"
ssh "root@$IP"

# On server:
export HF_TOKEN="<your-token>"
source /opt/hf-env/bin/activate
huggingface-cli login --token $HF_TOKEN

# Upload README first
huggingface-cli upload schismaudio/e-gmd /root/e-gmd.md README.md --repo-type dataset

# Run the upload
bash /root/hcloud_egmd.sh
```

**Step 4: Verify and tear down**

```bash
# Verify on HF
# Then from local:
hcloud server delete egmd-upload
```

**Step 5: Commit**

```bash
git add scripts/hf_upload/cards/e-gmd.md scripts/hf_upload/hcloud_egmd.sh
git commit -m "feat: upload E-GMD to schismaudio/e-gmd via hcloud"
```

---

### Task 13: Upload Slakh2100 via hcloud

**Files:**
- Create: `scripts/hf_upload/cards/slakh2100.md`
- Create: `scripts/hf_upload/hcloud_slakh.sh`

**Step 1: Write Slakh2100 dataset card**

- **License:** CC-BY 4.0
- **Citation:** Manilow, E., Wichern, G., Seetharaman, P., & Le Roux, J. (2019). "Cutting Music Source Separation Some Slakh." ISMIR 2019
- **Description:** 2,100 synthesized multi-instrument tracks (~145 hours). Every track has isolated stems + MIDI for 34 instrument classes. Drums are MIDI program 128. Synthesized from Lakh MIDI using randomized Native Instruments patches.
- **Tags:** `music-transcription`, `source-separation`, `multi-instrument`, `midi`, `synthesized`
- **Size:** ~105GB download, ~500GB WAV

**Step 2: Write hcloud upload script**

```bash
#!/usr/bin/env bash
# Download Slakh2100 from Zenodo and upload to HF.
set -euo pipefail

source /opt/hf-env/bin/activate
cd /root && mkdir -p slakh-work && cd slakh-work

echo "=== Downloading Slakh2100 from Zenodo ==="
# Zenodo record 4599666 — check exact file URLs
# May be multiple zip/tar parts
aria2c -x 16 "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1" \
    -o slakh2100.tar.gz

echo "=== Extracting ==="
tar xzf slakh2100.tar.gz
rm slakh2100.tar.gz

echo "=== Uploading to HF (large folder) ==="
python3 -c "
from huggingface_hub import upload_large_folder
upload_large_folder(
    repo_id='schismaudio/slakh2100',
    repo_type='dataset',
    folder_path='/root/slakh-work/slakh2100_flac_redux',
)
"

echo "=== Done ==="
```

**Step 3: Spin up CX52 (480GB NVMe), upload README, run script**

```bash
./scripts/hf_upload/hcloud_bootstrap.sh cx52 slakh-upload
```

**Step 4: Verify and tear down, commit**

---

### Task 14: Upload STAR Drums (Permissive Subset) via hcloud

**Depends on:** Task 10 (license audit must be complete)

**Files:**
- Create: `scripts/hf_upload/cards/star-drums.md`
- Create: `scripts/hf_upload/hcloud_star.sh`

**Step 1: Determine redistributable track list**

Use the results from Task 10's license audit (`star_license_audit_results.json`) to know which tracks to include.

**Step 2: Write dataset card**

- **License:** CC-BY 4.0 (for the redistributable subset)
- **Citation:** Weber, P., Uhle, C., Muller, M., & Lang, M. (2025). "STAR Drums: A Dataset for Automatic Drum Transcription." TISMIR 8(1)
- **Description:** Subset of STAR Drums containing only tracks with CC-BY or more permissive licenses. [N] tracks out of the full dataset. 18 drum classes with onset annotations.
- **Important:** Document clearly which tracks are included and why others are excluded. Link to original Zenodo for the full dataset.
- **Tags:** `drum-transcription`, `real-audio`, `accompaniment`

**Step 3: Write hcloud upload script**

Similar to E-GMD but with track filtering:

```bash
#!/usr/bin/env bash
# Download STAR Drums from Zenodo, filter to permissive tracks, upload.
set -euo pipefail

source /opt/hf-env/bin/activate
cd /root && mkdir -p star-work && cd star-work

echo "=== Downloading STAR Drums from Zenodo ==="
# Zenodo record 15690078 — 6 parts x ~32GB each
# Download only what's needed based on license audit
# Adapt this after the audit determines which parts contain permissive tracks

echo "=== Filtering to permissive tracks ==="
# Use the audit results to copy only redistributable tracks
# python3 filter_star.py --audit-results star_license_audit_results.json

echo "=== Uploading ==="
huggingface-cli upload schismaudio/star-drums ./star-filtered/ --repo-type dataset
```

**Step 4: Spin up server, run, verify, tear down, commit**

---

### Task 15: Upload StemGMD via hcloud

The largest dataset — 1.13TB extracted. Requires special handling.

**Files:**
- Create: `scripts/hf_upload/cards/stemgmd.md`
- Create: `scripts/hf_upload/hcloud_stemgmd.sh`

**Step 1: Write StemGMD dataset card**

- **License:** CC-BY 4.0
- **Citation:** Ferroni et al. (2023). "StemGMD: A Large-Scale Multi-Kit Audio Dataset for Deep Drums Demixing."
- **Description:** 1,224 hours of audio. Built on GMD, rendered with 10 acoustic drum kits from Logic Pro X. Isolated single-instrument stems for a canonical 9-piece kit. Aligned MIDI from GMD.
- **Tags:** `drum-separation`, `multi-kit`, `source-separation`, `midi`
- **Size category:** `1M<n<10M`

**Step 2: Plan the stream-process approach**

StemGMD is split into multiple Zenodo archives. Download one part, extract, upload, delete, next:

```bash
#!/usr/bin/env bash
# Stream-process StemGMD: download/extract/upload one part at a time.
set -euo pipefail

source /opt/hf-env/bin/activate
cd /root && mkdir -p stemgmd-work && cd stemgmd-work

# Zenodo record 7860223 — check exact part URLs and count
PARTS=(
    "https://zenodo.org/records/7860223/files/StemGMD_part1.zip?download=1"
    "https://zenodo.org/records/7860223/files/StemGMD_part2.zip?download=1"
    # ... add all part URLs after checking Zenodo
)

for i in "${!PARTS[@]}"; do
    echo "=== Processing part $((i+1))/${#PARTS[@]} ==="
    aria2c -x 16 "${PARTS[$i]}" -o "part_${i}.zip"
    unzip "part_${i}.zip" -d "part_${i}"
    rm "part_${i}.zip"

    echo "=== Uploading part $((i+1)) ==="
    python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    repo_id='schismaudio/stemgmd',
    repo_type='dataset',
    folder_path='/root/stemgmd-work/part_${i}',
)
"
    rm -rf "part_${i}"
    echo "=== Part $((i+1)) done ==="
done
```

**Step 3: Spin up CCX53 or similar, run**

```bash
./scripts/hf_upload/hcloud_bootstrap.sh ccx53 stemgmd-upload
```

This will take 8-16 hours. Consider using `tmux` or `nohup` on the server.

**Step 4: Verify, tear down, commit**

---

### Task 16: Create Feature Pre-computation Script

Generate mel spectrograms and onset labels for the ADT datasets.

**Files:**
- Create: `scripts/hf_upload/compute_features.py`

**Step 1: Write the feature computation script**

```python
#!/usr/bin/env python3
"""Compute mel spectrograms and onset labels for ADT datasets on HF.

Usage:
    python scripts/hf_upload/compute_features.py \
        --dataset schismaudio/e-gmd \
        --output-dir /tmp/features/e-gmd \
        --n-mels 128 --sr 16000 --hop-length 512
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np


def compute_mel(audio_path: str, sr: int = 16000, n_mels: int = 128,
                hop_length: int = 512) -> np.ndarray:
    """Compute mel spectrogram for a single audio file."""
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def compute_onset_labels(annotation_path: str, n_frames: int,
                         sr: int = 16000, hop_length: int = 512,
                         n_classes: int = 26) -> np.ndarray:
    """Compute frame-level onset labels from annotation file.

    Returns: (n_frames, n_classes) binary array with target widening.
    """
    # TODO: Adapt parsing based on dataset format (MIDI for E-GMD/GMD,
    # TSV for STAR, etc.)
    labels = np.zeros((n_frames, n_classes), dtype=np.float32)
    # Parse annotations and fill in onset frames
    # Apply target widening: [0.3, 0.6, 1.0, 0.6, 0.3]
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-mels", type=int, default=128)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--hop-length", type=int, default=512)
    args = parser.parse_args()
    # Process all audio files in input-dir
    # Save as .npy or .parquet
```

**Step 2: Test locally on a small subset**

```bash
python scripts/hf_upload/compute_features.py \
    --input-dir ~/Documents/Datasets/e-gmd/drummer3/session1 \
    --output-dir /tmp/features-test
ls /tmp/features-test/
```

**Step 3: Commit**

```bash
git add scripts/hf_upload/compute_features.py
git commit -m "feat: add feature pre-computation script for ADT datasets"
```

---

### Task 17: Generate and Upload Features for ADT Datasets

**Depends on:** Tasks 4, 12, 13, 14, 15 (datasets must be uploaded), Task 16 (script)

For each ADT dataset (E-GMD, GMD, StemGMD, Slakh2100, STAR partial), run feature computation and upload as a separate config.

**Step 1: Run feature computation**

For small datasets (GMD), run locally. For large ones (E-GMD, StemGMD, Slakh2100), run on hcloud during or after the raw upload.

```bash
# GMD (local)
python scripts/hf_upload/compute_features.py \
    --input-dir /tmp/datasets/gmd/groove \
    --output-dir /tmp/features/gmd

# Upload features as separate directory
huggingface-cli upload schismaudio/groove-midi-dataset \
    /tmp/features/gmd features/ --repo-type dataset
```

**Step 2: Verify features config works**

```python
from datasets import load_dataset
ds = load_dataset("schismaudio/groove-midi-dataset", data_dir="features", split="train")
```

**Step 3: Repeat for all ADT datasets, commit any script updates**

---

### Task 18: Generate Visualization Images

Create visualization images for embedding in dataset cards.

**Files:**
- Create: `scripts/hf_upload/generate_visualizations.py`

**Step 1: Write visualization script**

Generate for each ADT dataset:
- Sample mel spectrogram with onset markers
- Class distribution histogram
- Duration distribution histogram
- Tempo/BPM distribution (if available)

```python
#!/usr/bin/env python3
"""Generate visualization images for HF dataset cards."""
import matplotlib.pyplot as plt
import numpy as np
# ... generate plots, save as PNG
```

**Step 2: Generate images and upload to each dataset repo**

```bash
# Upload images to repo
huggingface-cli upload schismaudio/e-gmd ./visualizations/ images/ --repo-type dataset
```

**Step 3: Update dataset cards to reference the images**

```markdown
## Visualizations

![Class Distribution](images/class_distribution.png)
![Sample Spectrogram](images/sample_spectrogram.png)
```

**Step 4: Commit**

```bash
git add scripts/hf_upload/generate_visualizations.py
git commit -m "feat: add visualization generation for dataset cards"
```

---

### Task 19: Create Collection Page & Cross-references

**Files:**
- Create: `scripts/hf_upload/cards/collection-description.md`

**Step 1: Write collection overview**

Create the description for the `schismaudio/drum-audio-datasets` collection:

```markdown
# Drum Audio Datasets

A comprehensive collection of drum and percussion datasets for machine learning research.
All datasets use permissive licenses (CC-BY 4.0, CC0, Apache 2.0).

## Use Cases

| Use Case | Recommended Datasets |
|----------|---------------------|
| **Drum Transcription (ADT)** | E-GMD, GMD, STAR Drums |
| **Drum Source Separation** | StemGMD, Slakh2100 |
| **Drum Synthesis / Generation** | WaivOps family, Dynamic Percussion, Drum & Perc Kits |
| **Audio Augmentation** | RIRS-NOISES, dEchorate |
| **Drum Classification** | Dynamic Percussion, VCSL Percussion |

## Dataset Comparison

| Dataset | Hours | Classes | Audio Type | License |
|---------|-------|---------|------------|---------|
| E-GMD | 444h | GM MIDI | Electronic | CC-BY 4.0 |
| GMD | 13.6h | 9 | Electronic | CC-BY 4.0 |
| StemGMD | 1,224h | 9 | Synthesized | CC-BY 4.0 |
| Slakh2100 | 145h | 34 (multi-inst) | Synthesized | CC-BY 4.0 |
| STAR Drums | TBD | 18 | Real + accompaniment | CC-BY 4.0 (subset) |
```

**Step 2: Update the HF collection**

Go to the collection settings and paste the description. Add all 16 dataset repos to the collection in the order listed in the design doc.

**Step 3: Update cross-references in all dataset cards**

For each dataset card, update the "Related Datasets" section to link to relevant other datasets in the collection.

**Step 4: Commit**

```bash
git add scripts/hf_upload/cards/collection-description.md
git commit -m "feat: add collection page and cross-references"
```

---

### Task 20: Final Verification Pass

**Step 1: Test load_dataset() for all 16 repos**

```python
import sys
from datasets import load_dataset

repos = [
    "schismaudio/e-gmd",
    "schismaudio/groove-midi-dataset",
    "schismaudio/stemgmd",
    "schismaudio/slakh2100",
    "schismaudio/star-drums",
    "schismaudio/dynamic-percussion",
    "schismaudio/drum-percussion-kits",
    "schismaudio/vcsl-percussion",
    "schismaudio/waivops-lofi-drums",
    "schismaudio/waivops-world-percussion",
    "schismaudio/waivops-edm-house",
    "schismaudio/waivops-retro-drums",
    "schismaudio/waivops-hiphop-lofi",
    "schismaudio/waivops-edm-tr808",
    "schismaudio/rirs-noises",
    "schismaudio/dechorate",
]

for repo in repos:
    try:
        ds = load_dataset(repo, streaming=True, split="train")
        sample = next(iter(ds))
        print(f"PASS: {repo} — keys: {list(sample.keys())}")
    except Exception as e:
        print(f"FAIL: {repo} — {e}", file=sys.stderr)
```

**Step 2: Visual review of all dataset cards**

Visit each repo on HF and check:
- README renders correctly
- Audio preview works (for datasets with audio columns)
- File browser shows correct structure
- License tag is correct
- Collection page lists all datasets

**Step 3: Check collection page**

Visit https://huggingface.co/collections/schismaudio/drum-audio-datasets and verify:
- All 16 datasets are listed
- Overview and comparison table render correctly
- Cross-references work

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete HF drum dataset ecosystem — 16 datasets uploaded"
```

---

## Task Dependency Graph

```
Task 1 (Setup) ──────────────────────────────┐
Task 2 (Card template) ──────────────────────┤
Task 3 (Upload script) ──────────────────────┤
                                              ▼
                                  ┌── Task 4 (GMD)
                                  ├── Task 5 (Dynamic Perc)
                                  ├── Task 6 (Drum Kits)
                                  ├── Task 7 (VCSL)
                                  ├── Task 8 (WaivOps x6)
                                  ├── Task 9 (RIRs)
                                  │
Task 10 (STAR audit) ────────────┤
Task 11 (hcloud script) ────────┤
                                  ├── Task 12 (E-GMD hcloud)
                                  ├── Task 13 (Slakh hcloud)
                                  ├── Task 14 (STAR hcloud) ← depends on Task 10
                                  ├── Task 15 (StemGMD hcloud)
                                  │
Task 16 (Feature script) ───────┤
                                  ├── Task 17 (Features upload) ← depends on 4,12-15,16
                                  ├── Task 18 (Visualizations) ← depends on 4-15
                                  │
                                  ├── Task 19 (Collection page) ← depends on 4-15
                                  └── Task 20 (Final verification) ← depends on all
```

## Parallelization Opportunities

These groups can run in parallel:
- **Group A (local):** Tasks 4-9 (all local uploads, independent)
- **Group B (hcloud):** Tasks 12, 13, 15 (can run on separate servers simultaneously)
- **Group C (after uploads):** Tasks 17, 18, 19 (independent of each other, depend on uploads)

Task 10 (STAR audit) should start early since Task 14 depends on it.
Task 11 (hcloud script) should be done before any hcloud uploads.
