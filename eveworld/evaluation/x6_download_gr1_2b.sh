#!/usr/bin/env bash
set -euo pipefail

# Download open-gigaai/GigaWorld-0-Video-GR1-2b (DreamGen eval); resumable, rerun after an interruption.
# text_encoder (t5-11b) and VAE already exist locally, so they are not re-downloaded.

ROOT_DIR="${ROOT_DIR:-${GAGI_ROOT:-$HOME/gagi}}"
DEST="${DEST:-${ROOT_DIR}/gigaworld0_gr1_2b}"
REPO_ID="open-gigaai/GigaWorld-0-Video-GR1-2b"

export HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
export HF_XET_CACHE="${HF_XET_CACHE:-${ROOT_DIR}/.hf_xet_cache}"

source "${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-gigaworld}"

python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="${REPO_ID}", local_dir="${DEST}", max_workers=8)
print("download done:", p)
PY

echo "== layout probe =="
find "${DEST}" -maxdepth 2 -name "config.json" -o -maxdepth 2 -name "*.safetensors" | head -20
echo "done; copied to: ${DEST}"
