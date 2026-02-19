#!/usr/bin/env python3
"""STAR Drums License Audit

Audits the STAR Drums dataset to identify which tracks have permissive
(redistributable) vs. restrictive (non-commercial) licenses.

The STAR dataset contains tracks from three sources:
  1. ISMIR 2004 Genre Dataset - all CC-BY-NC-SA 1.0 (restrictive)
  2. MUSDB18 - all CC-BY-NC-SA 4.0 (restrictive), with two tracks at 3.0
  3. MTG-Jamendo - mixed licenses, per-track (see licenses_audio_files_from_mtg_jamendo.txt)

Output: scripts/hf_upload/star_license_audit_results.json
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

# -- Configuration --
STAR_ROOT = Path.home() / "Documents" / "Datasets" / "star-drums"
DATA_ROOT = STAR_ROOT / "data"
LICENSE_DIR = STAR_ROOT / "LICENSE"
OUTPUT_PATH = Path(__file__).parent / "star_license_audit_results.json"

# License classification: anything with "nc" (non-commercial) is restrictive
PERMISSIVE_KEYWORDS = {"by", "by-sa"}  # CC-BY and CC-BY-SA variants
RESTRICTIVE_KEYWORDS = {"by-nc", "by-nc-sa", "by-nc-nd"}  # Any NC variant


def classify_license(url: str) -> str:
    """Classify a CC license URL as 'permissive' or 'restrictive'.

    Permissive: CC-BY, CC-BY-SA, CC0 (allows commercial use and redistribution)
    Restrictive: CC-BY-NC, CC-BY-NC-SA, CC-BY-NC-ND (non-commercial restrictions)
    """
    url_lower = url.lower()
    if "/by-nc" in url_lower:
        return "restrictive"
    elif "/by-sa" in url_lower or "/by/" in url_lower:
        return "permissive"
    elif "cc0" in url_lower or "publicdomain" in url_lower:
        return "permissive"
    else:
        return "unknown"


def normalize_license(url: str) -> str:
    """Normalize a CC license URL to a canonical form for grouping.

    Strips locale suffixes and version differences for summary purposes.
    E.g., 'http://creativecommons.org/licenses/by-nc-sa/3.0/nl/' -> 'CC-BY-NC-SA'
    """
    match = re.search(r"/licenses/([\w-]+)/", url)
    if match:
        return "CC-" + match.group(1).upper()
    return url


def parse_mtg_jamendo_licenses() -> dict[str, dict]:
    """Parse the MTG-Jamendo per-track license file.

    Returns dict mapping track_id -> {license_url, license_name, classification}
    """
    license_file = LICENSE_DIR / "licenses_audio_files_from_mtg_jamendo.txt"
    with open(license_file, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    track_licenses = {}

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Line 1: "8559.mp3"
        track_id = lines[0].replace(".mp3", "").strip()

        # Line 3: "Available under a Creative Commons ... license: http://..."
        license_match = re.search(
            r"(https?://creativecommons\.org/licenses/[^\s]+)", lines[2]
        )
        if license_match:
            url = license_match.group(1)
            track_licenses[track_id] = {
                "license_url": url,
                "license_normalized": normalize_license(url),
                "classification": classify_license(url),
            }

    return track_licenses


def get_annotation_files(split_dir: Path) -> list[str]:
    """Get all annotation filenames from a split directory.

    Handles both flat structure (validation/test) and nested structure (training).
    """
    ann_dir = split_dir / "annotation"
    if ann_dir.exists():
        return sorted(os.listdir(ann_dir))

    # Training has subdirectories (ismir04, mtg-jamendo)
    files = []
    for subdir in sorted(split_dir.iterdir()):
        sub_ann = subdir / "annotation"
        if sub_ann.exists():
            for f in sorted(os.listdir(sub_ann)):
                files.append(f"{subdir.name}/{f}")
    return files


def audit_tracks() -> dict:
    """Run the full license audit across all STAR Drums tracks.

    Returns a structured results dictionary.
    """
    # Parse MTG-Jamendo per-track licenses
    jamendo_licenses = parse_mtg_jamendo_licenses()

    results = {
        "metadata": {
            "dataset": "STAR Drums",
            "dataset_path": str(STAR_ROOT),
            "sources": {
                "ismir04": {
                    "description": "ISMIR 2004 Genre Dataset",
                    "license": "CC-BY-NC-SA 1.0",
                    "license_url": "https://creativecommons.org/licenses/by-nc-sa/1.0/",
                    "classification": "restrictive",
                },
                "musdb18": {
                    "description": "MUSDB18 (validation + test splits)",
                    "license": "CC-BY-NC-SA 4.0 (most), CC-BY-NC-SA 3.0 (2 tracks)",
                    "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                    "classification": "restrictive",
                },
                "mtg-jamendo": {
                    "description": "MTG-Jamendo Dataset",
                    "license": "Mixed (per-track)",
                    "classification": "mixed",
                },
            },
        },
        "summary": {},
        "tracks": {
            "training": {"ismir04": [], "mtg-jamendo": []},
            "validation": [],
            "test": [],
        },
    }

    all_tracks = []
    license_counts = Counter()
    classification_counts = Counter()
    source_counts = defaultdict(lambda: {"permissive": 0, "restrictive": 0, "total": 0})

    # --- Training: ISMIR 2004 (all CC-BY-NC-SA 1.0 = restrictive) ---
    ismir_ann_dir = DATA_ROOT / "training" / "ismir04" / "annotation"
    if ismir_ann_dir.exists():
        for f in sorted(os.listdir(ismir_ann_dir)):
            track_info = {
                "filename": f,
                "source": "ismir04",
                "split": "training",
                "license_url": "https://creativecommons.org/licenses/by-nc-sa/1.0/",
                "license_normalized": "CC-BY-NC-SA",
                "classification": "restrictive",
            }
            results["tracks"]["training"]["ismir04"].append(track_info)
            all_tracks.append(track_info)
            license_counts["CC-BY-NC-SA 1.0"] += 1
            classification_counts["restrictive"] += 1
            source_counts["ismir04"]["restrictive"] += 1
            source_counts["ismir04"]["total"] += 1

    # --- Training: MTG-Jamendo (per-track licenses) ---
    jamendo_ann_dir = DATA_ROOT / "training" / "mtg-jamendo" / "annotation"
    if jamendo_ann_dir.exists():
        for f in sorted(os.listdir(jamendo_ann_dir)):
            # Extract track ID (numeric prefix before first _mix_)
            track_id = f.split("_")[0]
            lic_info = jamendo_licenses.get(track_id, {})
            classification = lic_info.get("classification", "unknown")
            license_norm = lic_info.get("license_normalized", "UNKNOWN")
            license_url = lic_info.get("license_url", "")

            track_info = {
                "filename": f,
                "source": "mtg-jamendo",
                "split": "training",
                "track_id": track_id,
                "license_url": license_url,
                "license_normalized": license_norm,
                "classification": classification,
            }
            results["tracks"]["training"]["mtg-jamendo"].append(track_info)
            all_tracks.append(track_info)
            license_counts[license_norm] += 1
            classification_counts[classification] += 1
            source_counts["mtg-jamendo"][classification] += 1
            source_counts["mtg-jamendo"]["total"] += 1

    # --- Validation: MUSDB18 (all CC-BY-NC-SA 4.0, except 2 at 3.0) ---
    val_ann_dir = DATA_ROOT / "validation" / "annotation"
    nc_sa_3_tracks = {
        "The_Easton_Ellises__Falcon_69",
        "The_Easton_Ellises_(Baumi)__SDRNR",
    }
    if val_ann_dir.exists():
        for f in sorted(os.listdir(val_ann_dir)):
            base_name = f.split("_mix_")[0]
            if base_name in nc_sa_3_tracks:
                lic_url = "https://creativecommons.org/licenses/by-nc-sa/3.0/"
                lic_ver = "CC-BY-NC-SA 3.0"
            else:
                lic_url = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
                lic_ver = "CC-BY-NC-SA 4.0"

            track_info = {
                "filename": f,
                "source": "musdb18",
                "split": "validation",
                "license_url": lic_url,
                "license_normalized": "CC-BY-NC-SA",
                "classification": "restrictive",
            }
            results["tracks"]["validation"].append(track_info)
            all_tracks.append(track_info)
            license_counts[lic_ver] += 1
            classification_counts["restrictive"] += 1
            source_counts["musdb18"]["restrictive"] += 1
            source_counts["musdb18"]["total"] += 1

    # --- Test: MUSDB18 (same licensing) ---
    test_ann_dir = DATA_ROOT / "test" / "annotation"
    if test_ann_dir.exists():
        for f in sorted(os.listdir(test_ann_dir)):
            base_name = f.split("_mix_")[0]
            if base_name in nc_sa_3_tracks:
                lic_url = "https://creativecommons.org/licenses/by-nc-sa/3.0/"
                lic_ver = "CC-BY-NC-SA 3.0"
            else:
                lic_url = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
                lic_ver = "CC-BY-NC-SA 4.0"

            track_info = {
                "filename": f,
                "source": "musdb18",
                "split": "test",
                "license_url": lic_url,
                "license_normalized": "CC-BY-NC-SA",
                "classification": "restrictive",
            }
            results["tracks"]["test"].append(track_info)
            all_tracks.append(track_info)
            license_counts[lic_ver] += 1
            classification_counts["restrictive"] += 1
            source_counts["musdb18"]["restrictive"] += 1
            source_counts["musdb18"]["total"] += 1

    # --- Build summary ---
    total = len(all_tracks)
    permissive = classification_counts.get("permissive", 0)
    restrictive = classification_counts.get("restrictive", 0)
    unknown = classification_counts.get("unknown", 0)

    results["summary"] = {
        "total_tracks": total,
        "permissive_count": permissive,
        "restrictive_count": restrictive,
        "unknown_count": unknown,
        "permissive_pct": round(100 * permissive / total, 1) if total > 0 else 0,
        "restrictive_pct": round(100 * restrictive / total, 1) if total > 0 else 0,
        "license_distribution": dict(
            sorted(license_counts.items(), key=lambda x: -x[1])
        ),
        "by_source": {
            src: dict(counts) for src, counts in sorted(source_counts.items())
        },
        "by_split": {
            "training_ismir04": len(results["tracks"]["training"]["ismir04"]),
            "training_mtg_jamendo": len(results["tracks"]["training"]["mtg-jamendo"]),
            "validation": len(results["tracks"]["validation"]),
            "test": len(results["tracks"]["test"]),
        },
        "recommendation": "",
    }

    # Generate recommendation
    if permissive / total >= 0.10:
        results["summary"]["recommendation"] = (
            f"Redistributable subset ({permissive} tracks, {results['summary']['permissive_pct']}%) "
            f"is large enough to host on HuggingFace. Upload only permissive-licensed tracks."
        )
    else:
        results["summary"]["recommendation"] = (
            f"Redistributable subset ({permissive} tracks, {results['summary']['permissive_pct']}%) "
            f"is too small (<10%). Consider skipping STAR or hosting annotations only."
        )

    # --- Permissive track list (for easy filtering) ---
    results["permissive_tracks"] = [
        {
            "filename": t["filename"],
            "source": t["source"],
            "split": t["split"],
            "license": t["license_normalized"],
        }
        for t in all_tracks
        if t["classification"] == "permissive"
    ]

    return results


def print_report(results: dict) -> None:
    """Print a human-readable summary of the audit results."""
    s = results["summary"]

    print("=" * 70)
    print("STAR Drums License Audit Report")
    print("=" * 70)
    print()
    print(f"Dataset path: {results['metadata']['dataset_path']}")
    print()
    print("--- Track Counts ---")
    print(f"  Total tracks:       {s['total_tracks']:>6}")
    print(f"  Permissive:         {s['permissive_count']:>6}  ({s['permissive_pct']}%)")
    print(f"  Restrictive (NC):   {s['restrictive_count']:>6}  ({s['restrictive_pct']}%)")
    if s["unknown_count"]:
        print(f"  Unknown:            {s['unknown_count']:>6}")
    print()

    print("--- By Source ---")
    for src, counts in sorted(s["by_source"].items()):
        perm = counts.get("permissive", 0)
        rest = counts.get("restrictive", 0)
        tot = counts.get("total", 0)
        print(f"  {src:<15} total={tot:>5}  permissive={perm:>5}  restrictive={rest:>5}")
    print()

    print("--- By Split ---")
    for split, count in s["by_split"].items():
        print(f"  {split:<25} {count:>5}")
    print()

    print("--- License Distribution ---")
    for lic, count in s["license_distribution"].items():
        print(f"  {count:>5}  {lic}")
    print()

    print("--- Recommendation ---")
    print(f"  {s['recommendation']}")
    print()
    print("=" * 70)


def main():
    results = audit_tracks()

    # Write JSON output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to: {OUTPUT_PATH}")
    print()

    # Print report
    print_report(results)


if __name__ == "__main__":
    main()
