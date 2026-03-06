"""Monitor training checkpoints and evaluate F1 as they appear."""
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path("outputs")
EVALUATED = set()
POLL_INTERVAL = 120  # check every 2 minutes


def find_checkpoints():
    """Find all checkpoint_epochN.pt files, sorted by epoch."""
    ckpts = sorted(
        OUTPUT_DIR.glob("checkpoint_epoch*.pt"),
        key=lambda p: int(p.stem.split("epoch")[1]),
    )
    return ckpts


def evaluate(ckpt_path):
    """Run validate.py on a checkpoint and return the output."""
    cmd = [
        sys.executable, "scripts/validate.py",
        str(ckpt_path),
        "--datasets", "e-gmd-aug", "star-drums-aug",
        "--max-batches", "10",
        "--batch-size", "32",
        "--threshold", "0.3",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return result.stdout + result.stderr


def main():
    print("F1 Monitor started. Watching for checkpoints...")
    print(f"Poll interval: {POLL_INTERVAL}s")
    print("=" * 60)

    # Mark already-evaluated checkpoints
    for ckpt in find_checkpoints():
        epoch = int(ckpt.stem.split("epoch")[1])
        if epoch <= 20:
            EVALUATED.add(ckpt.name)
            print(f"Skipping {ckpt.name} (already evaluated)")

    while True:
        for ckpt in find_checkpoints():
            if ckpt.name in EVALUATED:
                continue

            epoch = int(ckpt.stem.split("epoch")[1])
            print(f"\nNew checkpoint: {ckpt.name}")
            print(f"Evaluating epoch {epoch}...")

            output = evaluate(ckpt)

            # Extract results
            for line in output.split("\n"):
                if any(k in line for k in ["Precision:", "Recall:", "F1:", "Reference", "Estimated", "Validation Results"]):
                    print(line)

            EVALUATED.add(ckpt.name)
            print("-" * 60)
            sys.stdout.flush()

        # Check if training is done (final.pt exists)
        if (OUTPUT_DIR / "final.pt").exists():
            print("\nfinal.pt detected — training complete!")
            break

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
