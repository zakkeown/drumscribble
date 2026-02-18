#!/usr/bin/env bash
# Upload E-GMD to HF Hub as tar archives to avoid the 10k-files-per-directory limit.
# Tars one drummer/session at a time to fit in available disk space.
set -euo pipefail

EGMD_DIR="${1:-$HOME/Documents/Datasets/e-gmd}"
REPO="zkeown/e-gmd-v1"
TMP="/tmp/egmd-upload"

mkdir -p "$TMP"
cd "$EGMD_DIR"

echo "=== Uploading metadata files ==="
hf upload "$REPO" e-gmd-v1.0.0.csv e-gmd-v1.0.0.csv --repo-type dataset
hf upload "$REPO" LICENSE LICENSE --repo-type dataset
hf upload "$REPO" README README --repo-type dataset

# Small drummers (< 30GB): tar the whole directory
for d in drummer3 drummer4 drummer6 drummer7 drummer8 drummer9 drummer10; do
    echo "=== Tarring $d ==="
    tar cf "$TMP/${d}.tar" "$d"
    echo "=== Uploading ${d}.tar ==="
    hf upload "$REPO" "$TMP/${d}.tar" "${d}.tar" --repo-type dataset
    rm "$TMP/${d}.tar"
    echo "=== Done: $d ==="
done

# Large drummers (> 30GB): tar each session separately
for d in drummer1 drummer5; do
    for session_dir in "$d"/session*/; do
        sname=$(basename "$session_dir")
        tarname="${d}_${sname}.tar"
        echo "=== Tarring $d/$sname ==="
        tar cf "$TMP/$tarname" "$session_dir"
        echo "=== Uploading $tarname ==="
        hf upload "$REPO" "$TMP/$tarname" "$tarname" --repo-type dataset
        rm "$TMP/$tarname"
        echo "=== Done: $d/$sname ==="
    done
done

rmdir "$TMP" 2>/dev/null || true
echo ""
echo "All done! Dataset at: https://huggingface.co/datasets/$REPO"
