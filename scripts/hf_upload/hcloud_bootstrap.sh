#!/usr/bin/env bash
# Bootstrap a Hetzner Cloud server for HF dataset uploads.
#
# Usage:
#   ./scripts/hf_upload/hcloud_bootstrap.sh <server-type> <server-name>
#
# Examples:
#   ./scripts/hf_upload/hcloud_bootstrap.sh cx32 egmd-upload
#   ./scripts/hf_upload/hcloud_bootstrap.sh cx52 slakh-upload
#   ./scripts/hf_upload/hcloud_bootstrap.sh ccx53 stemgmd-upload
#
# Server types and NVMe sizes:
#   cx22  = 40GB   (~$0.01/hr)
#   cx32  = 160GB  (~$0.02/hr)
#   cx42  = 320GB  (~$0.04/hr)
#   cx52  = 480GB  (~$0.07/hr)
#   ccx13 = 80GB   (~$0.02/hr)
#   ccx23 = 160GB  (~$0.04/hr)
#   ccx33 = 320GB  (~$0.07/hr)
#   ccx43 = 640GB  (~$0.13/hr)
#   ccx53 = 960GB  (~$0.20/hr)
#
# Prerequisites:
#   - hcloud CLI installed and configured
#   - SSH key named 'default' registered with Hetzner
set -euo pipefail

SERVER_TYPE="${1:?Usage: $0 <server-type> <server-name>}"
SERVER_NAME="${2:?Usage: $0 <server-type> <server-name>}"

echo "=== Creating server $SERVER_NAME (type: $SERVER_TYPE) ==="
hcloud server create \
    --type "$SERVER_TYPE" \
    --image ubuntu-24.04 \
    --name "$SERVER_NAME" \
    --ssh-key default \
    --location fsn1

IP=$(hcloud server ip "$SERVER_NAME")
echo "Server IP: $IP"

echo "Waiting for SSH to become available..."
for i in $(seq 1 30); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "root@$IP" true 2>/dev/null; then
        echo "SSH ready."
        break
    fi
    sleep 2
done

echo "=== Installing dependencies ==="
ssh -o StrictHostKeyChecking=no "root@$IP" bash <<'REMOTE'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq python3-pip python3-venv unzip aria2 tmux

python3 -m venv /opt/hf-env
source /opt/hf-env/bin/activate
pip install -q huggingface_hub[cli] datasets

echo ""
echo "=== Setup complete ==="
echo "Activate env: source /opt/hf-env/bin/activate"
echo "Login to HF:  huggingface-cli login"
REMOTE

echo ""
echo "========================================="
echo "Server ready!"
echo "========================================="
echo ""
echo "  SSH:          ssh root@$IP"
echo "  Activate:     source /opt/hf-env/bin/activate"
echo "  HF Login:     huggingface-cli login"
echo ""
echo "  When done:    hcloud server delete $SERVER_NAME"
echo ""
