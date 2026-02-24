#!/usr/bin/env bash
# RunPod training bootstrap for DrumscribbleCNN.
#
# Usage:
#   1. SSH into RunPod pod (A100 80GB SXM, PyTorch template, network volume at /workspace)
#   2. Clone repo:  git clone https://github.com/zakkeown/drumscribble.git /workspace/drumscribble
#   3. Run:         bash /workspace/drumscribble/scripts/runpod_train.sh
#
# Prerequisites:
#   - HF_TOKEN set in RunPod pod environment (or export it before running)
#   - Network volume mounted at /workspace
set -euo pipefail

# --- Verify environment ---
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export it or set it in RunPod pod template."
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi

# --- Install uv ---
if ! command -v uv &> /dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "uv: $(uv --version)"

# --- Install huggingface-cli with xet support ---
echo "=== Installing huggingface_hub[hf_xet] ==="
uv tool install "huggingface_hub[hf_xet]" --force

# --- Pre-cache HF shards to persistent volume ---
# Setting HF_HOME to /workspace ensures cache persists across pod restarts.
export HF_HOME=/workspace/hf_cache

echo ""
echo "=== Downloading dataset shards ==="
echo "This may take 30-60 min on first run (cache hits on subsequent runs)."
echo ""

echo "--- E-GMD augmented (train) ---"
huggingface-cli download schismaudio/e-gmd-aug --repo-type dataset \
    --include "features/*" --local-dir /workspace/data/e-gmd-aug

echo "--- STAR augmented (train) ---"
huggingface-cli download schismaudio/star-drums-aug --repo-type dataset \
    --include "features/*" --local-dir /workspace/data/star-drums-aug

echo "--- E-GMD (validation only) ---"
huggingface-cli download schismaudio/e-gmd --repo-type dataset \
    --include "features/validation-*" --local-dir /workspace/data/e-gmd-val

echo "--- STAR (validation only) ---"
huggingface-cli download zkeown/star-drums --repo-type dataset \
    --include "features/validation-*" --local-dir /workspace/data/star-drums-val

echo ""
echo "=== Download complete ==="
echo ""

# --- Launch training ---
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

TRAIN_CMD="export HF_TOKEN='${HF_TOKEN}' && \
    export HF_HOME=/workspace/hf_cache && \
    export PATH=\"\$HOME/.local/bin:\$PATH\" && \
    cd ${REPO_DIR} && \
    uv run scripts/train_hf_job.py \
        --dataset multi \
        --epochs 200 \
        --lr 1e-4 \
        --finetune \
        --num-workers 2 \
        --egmd-repo schismaudio/e-gmd-aug \
        --egmd-val-repo schismaudio/e-gmd \
        --star-repo schismaudio/star-drums-aug \
        --star-val-repo zkeown/star-drums \
        --resume-from final.pt \
    2>&1 | tee /workspace/train.log"

if command -v tmux &> /dev/null; then
    tmux new-session -d -s train "$TRAIN_CMD"
    echo "=== Training started in tmux session 'train' ==="
    echo ""
    echo "  Attach:   tmux attach -t train"
    echo "  Detach:   Ctrl+B, D"
    echo "  Logs:     tail -f /workspace/train.log"
else
    echo "=== Starting training (no tmux — running directly) ==="
    eval "$TRAIN_CMD"
fi
