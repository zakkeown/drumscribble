"""Remap STAR feature shards from 18-class to 26-class GM taxonomy.

Reads existing feature shards, remaps onset/velocity targets using
drumscribble's STAR-to-GM mapping with onset widening, and writes
new shards in-place.

Usage:
    uv run python scripts/remap_star_shards.py \
        --shard-root ~/Documents/Datasets/build/star-drums
"""
import argparse
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from drumscribble.config import NUM_CLASSES
from drumscribble.data.remap import remap_star_targets


def remap_shard(src_path: Path, dst_path: Path) -> int:
    """Remap one feature shard tar file.

    Returns the number of samples processed.
    """
    count = 0
    with tarfile.open(src_path, "r") as src, tarfile.open(dst_path, "w") as dst:
        members = src.getmembers()

        # Group members by sample key
        samples: dict[str, dict[str, tarfile.TarInfo]] = {}
        for m in members:
            # Split on first dot to get sample key
            dot_idx = m.name.index(".")
            key = m.name[:dot_idx]
            suffix = m.name[dot_idx + 1:]
            if key not in samples:
                samples[key] = {}
            samples[key][suffix] = m

        for key, parts in samples.items():
            # Read existing arrays
            onset_f = src.extractfile(parts["onset_targets.npy"])
            onset_18 = np.load(io.BytesIO(onset_f.read()))

            vel_f = src.extractfile(parts["velocity_targets.npy"])
            vel_18 = np.load(io.BytesIO(vel_f.read()))

            # Skip if already remapped
            if onset_18.shape[0] == NUM_CLASSES:
                # Copy all members unchanged
                for suffix, member in parts.items():
                    f = src.extractfile(member)
                    dst.addfile(member, f)
                count += 1
                continue

            # Remap
            onset_26, vel_26 = remap_star_targets(onset_18, vel_18)

            # Write mel unchanged
            mel_member = parts["mel_spectrogram.npy"]
            mel_f = src.extractfile(mel_member)
            dst.addfile(mel_member, mel_f)

            # Write remapped onset
            buf = io.BytesIO()
            np.save(buf, onset_26)
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{key}.onset_targets.npy")
            info.size = buf.getbuffer().nbytes
            dst.addfile(info, buf)

            # Write remapped velocity
            buf = io.BytesIO()
            np.save(buf, vel_26)
            buf.seek(0)
            info = tarfile.TarInfo(name=f"{key}.velocity_targets.npy")
            info.size = buf.getbuffer().nbytes
            dst.addfile(info, buf)

            # Write updated params
            params_f = src.extractfile(parts["params.json"])
            params = json.loads(params_f.read())
            params["n_classes"] = NUM_CLASSES
            params_bytes = json.dumps(params).encode()
            info = tarfile.TarInfo(name=f"{key}.params.json")
            info.size = len(params_bytes)
            dst.addfile(info, io.BytesIO(params_bytes))

            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Remap STAR feature shards from 18-class to 26-class GM"
    )
    parser.add_argument(
        "--shard-root",
        type=str,
        required=True,
        help="Root of STAR dataset (e.g. ~/Documents/Datasets/build/star-drums)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying files",
    )
    args = parser.parse_args()

    root = Path(args.shard_root).expanduser()
    features_dir = root / "data" / "features"

    for split in ["train", "validation", "test"]:
        split_dir = features_dir / split
        if not split_dir.exists():
            print(f"Skipping {split}: {split_dir} not found")
            continue

        shards = sorted(split_dir.glob("feature-shard-*.tar"))
        if not shards:
            print(f"Skipping {split}: no shards found")
            continue

        print(f"{split}: {len(shards)} shards")

        if args.dry_run:
            continue

        # Write to temp dir, then swap
        tmp_dir = Path(tempfile.mkdtemp(dir=split_dir.parent))
        try:
            for shard_path in shards:
                dst_path = tmp_dir / shard_path.name
                count = remap_shard(shard_path, dst_path)
                print(f"  {shard_path.name}: {count} samples remapped")

            # Verify all output shards exist and have content
            for shard_path in shards:
                dst_path = tmp_dir / shard_path.name
                assert dst_path.exists(), f"Missing output: {dst_path}"
                assert dst_path.stat().st_size > 0, f"Empty output: {dst_path}"

            # Swap: move originals out, move new in
            backup_dir = Path(tempfile.mkdtemp(dir=split_dir.parent))
            for shard_path in shards:
                shutil.move(str(shard_path), str(backup_dir / shard_path.name))
            for shard_path in shards:
                shutil.move(str(tmp_dir / shard_path.name), str(shard_path))

            # Clean up backup
            shutil.rmtree(backup_dir)
            print(f"  {split} done!")

        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    print("All splits remapped.")


if __name__ == "__main__":
    main()
