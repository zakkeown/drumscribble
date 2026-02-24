#!/usr/bin/env bash
# Launch a RunPod training pod via the REST API.
#
# Creates a network volume (if needed) and an A100 80GB SXM pod with
# the repo cloned and bootstrap script ready to run.
#
# Usage:
#   export RUNPOD_API_KEY=rpa_...
#   export HF_TOKEN=hf_...
#   bash scripts/runpod_launch.sh
#
# After the pod is running:
#   Connect via RunPod web terminal or SSH
#   bash /workspace/drumscribble/scripts/runpod_train.sh
#
# To stop/terminate later:
#   bash scripts/runpod_launch.sh stop <pod-id>
#   bash scripts/runpod_launch.sh terminate <pod-id>
set -euo pipefail

API="https://rest.runpod.io/v1"
VOLUME_NAME="drumscribble-data"
VOLUME_SIZE=50
DATA_CENTER="US-KS-2"
GPU_TYPE="NVIDIA A100-SXM4-80GB"
POD_NAME="drumscribble-train"
IMAGE="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
CONTAINER_DISK=20

# --- Validate env ---
if [ -z "${RUNPOD_API_KEY:-}" ]; then
    echo "ERROR: RUNPOD_API_KEY not set"
    exit 1
fi
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set"
    exit 1
fi

auth_header="Authorization: Bearer ${RUNPOD_API_KEY}"
json_header="Content-Type: application/json"

# --- Helper: stop or terminate a pod ---
if [ "${1:-}" = "stop" ] && [ -n "${2:-}" ]; then
    echo "Stopping pod ${2}..."
    curl -s --request POST \
        --url "${API}/pods/${2}/stop" \
        --header "$auth_header" | python3 -m json.tool
    exit 0
fi

if [ "${1:-}" = "terminate" ] && [ -n "${2:-}" ]; then
    echo "Terminating pod ${2}..."
    curl -s --request DELETE \
        --url "${API}/pods/${2}" \
        --header "$auth_header" | python3 -m json.tool
    exit 0
fi

# --- Step 1: Find or create network volume ---
echo "=== Checking for existing network volume '${VOLUME_NAME}' ==="
VOLUMES=$(curl -s --request GET \
    --url "${API}/networkvolumes" \
    --header "$auth_header")

VOLUME_ID=$(echo "$VOLUMES" | python3 -c "
import json, sys
data = json.load(sys.stdin)
for v in data:
    if v.get('name') == '${VOLUME_NAME}':
        print(v['id'])
        sys.exit(0)
" 2>/dev/null || true)

if [ -n "$VOLUME_ID" ]; then
    echo "Found existing volume: ${VOLUME_ID}"
else
    echo "Creating network volume '${VOLUME_NAME}' (${VOLUME_SIZE} GB in ${DATA_CENTER})..."
    VOLUME_RESP=$(curl -s --request POST \
        --url "${API}/networkvolumes" \
        --header "$auth_header" \
        --header "$json_header" \
        --data "{
            \"name\": \"${VOLUME_NAME}\",
            \"size\": ${VOLUME_SIZE},
            \"dataCenterId\": \"${DATA_CENTER}\"
        }")
    VOLUME_ID=$(echo "$VOLUME_RESP" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
    echo "Created volume: ${VOLUME_ID}"
fi

# --- Step 2: Create pod (use Python to build JSON payload safely) ---
echo ""
echo "=== Creating pod '${POD_NAME}' ==="
echo "  GPU: ${GPU_TYPE}"
echo "  Image: ${IMAGE}"
echo "  Volume: ${VOLUME_ID}"

POD_RESP=$(python3 -c "
import json, subprocess, sys

payload = {
    'name': '${POD_NAME}',
    'imageName': '${IMAGE}',
    'gpuTypeIds': ['${GPU_TYPE}'],
    'gpuCount': 1,
    'containerDiskInGb': ${CONTAINER_DISK},
    'networkVolumeId': '${VOLUME_ID}',
    'env': {
        'HF_TOKEN': sys.stdin.read().strip(),
        'HF_HOME': '/workspace/hf_cache',
    },
    'ports': ['22/tcp'],
    'dockerStartCmd': [
        'bash', '-c',
        'apt-get update -qq && apt-get install -y -qq tmux git > /dev/null 2>&1 && '
        '[ ! -d /workspace/drumscribble ] && '
        'git clone https://github.com/zakkeown/drumscribble.git /workspace/drumscribble; '
        'sleep infinity'
    ],
}

result = subprocess.run(
    ['curl', '-s', '--request', 'POST',
     '--url', '${API}/pods',
     '--header', 'Authorization: Bearer ${RUNPOD_API_KEY}',
     '--header', 'Content-Type: application/json',
     '--data', json.dumps(payload)],
    capture_output=True, text=True,
)
print(result.stdout)
" <<< "${HF_TOKEN}")

POD_ID=$(echo "$POD_RESP" | python3 -c "
import json, sys
d = json.load(sys.stdin)
if isinstance(d, list):
    print('ERROR: ' + json.dumps(d))
else:
    print(d.get('id', 'ERROR: ' + json.dumps(d)))
")

if [[ "$POD_ID" == ERROR:* ]]; then
    echo "$POD_ID"
    exit 1
fi

echo ""
echo "========================================="
echo "Pod created: ${POD_ID}"
echo "========================================="
echo ""
echo "  Status:     https://www.runpod.io/console/pods"
echo "  Pod ID:     ${POD_ID}"
echo "  Volume ID:  ${VOLUME_ID}"
echo ""
echo "  Once the pod is running:"
echo "    1. Connect via RunPod web terminal or SSH"
echo "    2. Run:  bash /workspace/drumscribble/scripts/runpod_train.sh"
echo "    3. Detach tmux: Ctrl+B, D"
echo ""
echo "  To stop (keeps volume, stops billing GPU):"
echo "    bash scripts/runpod_launch.sh stop ${POD_ID}"
echo ""
echo "  To terminate (destroys pod, keeps volume):"
echo "    bash scripts/runpod_launch.sh terminate ${POD_ID}"
echo ""
