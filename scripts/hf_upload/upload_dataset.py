#!/usr/bin/env python3
"""Upload a local dataset directory to HuggingFace Hub.

Usage:
    python scripts/hf_upload/upload_dataset.py \\
        --repo schismaudio/e-gmd \\
        --local-dir /tmp/datasets/e-gmd \\
        --readme /tmp/datasets/e-gmd/README.md

    # Large dataset with verification
    python scripts/hf_upload/upload_dataset.py \\
        --repo schismaudio/stemgmd \\
        --local-dir /tmp/datasets/stemgmd \\
        --verify

Features:
    - upload_folder for datasets <= 50GB
    - upload_large_folder with resume for datasets > 50GB
    - Optional verification via load_dataset() streaming
    - Progress reporting with total size before upload
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.2f} PB"


def _calculate_dir_size(path: Path) -> int:
    """Calculate total size of all files in a directory tree."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def upload_readme(repo_id: str, readme_path: Path) -> None:
    """Upload a README.md as the dataset card."""
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Uploading dataset card to {repo_id}...")
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print("  Dataset card uploaded.")


def upload_dataset(repo_id: str, local_dir: Path) -> None:
    """Upload a local directory to a HuggingFace dataset repo.

    Uses upload_folder for directories <= 50GB and upload_large_folder
    (with built-in resume support) for directories > 50GB.
    """
    from huggingface_hub import HfApi

    api = HfApi()

    total_bytes = _calculate_dir_size(local_dir)
    total_gb = total_bytes / (1024**3)
    print(f"Total upload size: {_format_size(total_bytes)} ({total_gb:.2f} GB)")

    threshold_gb = 50
    if total_gb > threshold_gb:
        print(f"Size exceeds {threshold_gb}GB — using upload_large_folder (resume-capable)...")
        from huggingface_hub import upload_large_folder

        upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
        )
    else:
        print("Using upload_folder...")
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(local_dir),
        )

    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"\nUpload complete: {url}")


def verify_dataset(repo_id: str) -> bool:
    """Verify the uploaded dataset by streaming the first sample."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Skipping verification: 'datasets' library not installed.")
        print("  Install with: pip install datasets")
        return False

    print(f"\nVerifying {repo_id} via load_dataset(streaming=True)...")
    try:
        ds = load_dataset(repo_id, split="train", streaming=True)
        sample = next(iter(ds))
        print(f"  Verification passed. Sample keys: {list(sample.keys())}")
        return True
    except StopIteration:
        print("  Verification warning: dataset loaded but appears empty.")
        return False
    except Exception as e:
        print(f"  Verification failed: {e}")
        print("  This may be expected if the dataset needs a custom loading script")
        print("  or does not have a 'train' split.")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload a local dataset directory to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s --repo schismaudio/e-gmd --local-dir ./data/e-gmd\n"
            "  %(prog)s --repo schismaudio/e-gmd --local-dir ./data/e-gmd "
            "--readme ./cards/e-gmd.md --verify\n"
        ),
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="HuggingFace repo ID, e.g. schismaudio/e-gmd",
    )
    parser.add_argument(
        "--local-dir",
        required=True,
        help="Local directory containing the dataset files to upload",
    )
    parser.add_argument(
        "--readme",
        default=None,
        help="Path to README.md to upload as the dataset card",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run load_dataset() verification after upload",
    )
    args = parser.parse_args()

    # Validate paths
    local_dir = Path(args.local_dir)
    if not local_dir.exists():
        print(f"Error: local directory does not exist: {local_dir}", file=sys.stderr)
        return 1
    if not local_dir.is_dir():
        print(f"Error: path is not a directory: {local_dir}", file=sys.stderr)
        return 1

    readme_path = None
    if args.readme:
        readme_path = Path(args.readme)
        if not readme_path.exists():
            print(f"Error: README file does not exist: {readme_path}", file=sys.stderr)
            return 1

    # Check that huggingface_hub is importable before starting
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print(
            "Error: huggingface_hub is not installed.\n"
            "  Install with: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    # Upload README first (so it's visible immediately on the repo page)
    if readme_path:
        upload_readme(args.repo, readme_path)

    # Upload dataset files
    upload_dataset(args.repo, local_dir)

    # Optional verification
    if args.verify:
        verify_dataset(args.repo)

    return 0


if __name__ == "__main__":
    sys.exit(main())
