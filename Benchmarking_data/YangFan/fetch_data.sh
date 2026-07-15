#!/usr/bin/env bash
#
# fetch_data.sh - download the TD-TSP raw datasets that are gitignored.
#
# These files are large binaries and are NOT tracked in git. Run this script
# once after cloning to populate data/tdtsp/ and testcases/:
#
#     bash Benchmarking_data/YangFan/fetch_data.sh
#
# It downloads:
#   1. The 12 city travel-time tensors (*.npy) from the NCO4TDTSP processed
#      dataset hosted on Google Drive (Yang & Fan, NeurIPS 2025).
#   2. The 20-node Beijing test set (beijing_20_dataset_10000.pt) from the
#      NCO4TDTSP GitHub repository.
#
# Sources:
#   - NCO4TDTSP repo:        https://github.com/Brelliothe/NCO4TDTSP
#   - Processed dataset:     https://drive.google.com/file/d/1JsWs0MwUXAyXLbD6VXeW4_GR8Utcuqz-/view
#
set -euo pipefail

# Resolve the directory this script lives in (so it works from any CWD).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GDRIVE_FILE_ID="1JsWs0MwUXAyXLbD6VXeW4_GR8Utcuqz-"
PT_URL="https://raw.githubusercontent.com/Brelliothe/NCO4TDTSP/main/testcases/beijing_20_dataset_10000.pt"
ARCHIVE="tdtsp_processed_dataset.zip"
NPY_DIR="data/tdtsp"

echo "==> Target directory: $SCRIPT_DIR"
mkdir -p "$NPY_DIR" testcases

# --- 1. City tensors (*.npy) from Google Drive -> data/tdtsp/ ----------------
if ls "$NPY_DIR"/*.npy >/dev/null 2>&1; then
    echo "==> City .npy tensors already present in $NPY_DIR; skipping Google Drive download."
else
    if ! command -v gdown >/dev/null 2>&1; then
        echo "==> 'gdown' not found; installing (pip install gdown)..."
        python3 -m pip install --quiet gdown
    fi
    echo "==> Downloading processed dataset from Google Drive..."
    gdown "https://drive.google.com/uc?id=${GDRIVE_FILE_ID}" -O "$ARCHIVE"

    echo "==> Extracting $ARCHIVE ..."
    # The archive packs the tensors under a tdtsp/ folder and also ships macOS
    # AppleDouble junk (__MACOSX/ and ._* resource forks). Exclude that junk,
    # then move the real *.npy files into data/tdtsp/.
    unzip -oq "$ARCHIVE" -x "__MACOSX/*" -d _extracted
    find _extracted -name '*.npy' ! -name '._*' -exec mv -f {} "$NPY_DIR"/ \;
    rm -rf _extracted "$ARCHIVE"
fi

# --- 2. Beijing 20-node test set (*.pt) from GitHub -> testcases/ ------------
if [ -f testcases/beijing_20_dataset_10000.pt ]; then
    echo "==> testcases/beijing_20_dataset_10000.pt already present; skipping."
else
    echo "==> Downloading beijing_20_dataset_10000.pt from GitHub..."
    curl -sSL -o testcases/beijing_20_dataset_10000.pt "$PT_URL"
fi

echo "==> Done. Contents of $SCRIPT_DIR:"
ls -1 "$NPY_DIR"/*.npy 2>/dev/null || echo "  (no .npy files found - check the archive layout)"
ls -1 testcases/*.pt 2>/dev/null || true
