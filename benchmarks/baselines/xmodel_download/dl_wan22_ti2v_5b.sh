#!/usr/bin/env bash
# Wan2.2-TI2V-5B (Alibaba, dense 5B, text+image to video). Not gated, ~10-12GB.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

REPO_ID="${REPO_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
DST="${DST:-${XMODEL_ROOT}/wan22_ti2v_5b}"

activate_env
check_login_if_gated 0

echo "==== [Wan2.2-TI2V-5B] starting download ===="
# diffusers weights only; skip raw .pth training state.
dl_repo "${REPO_ID}" "${DST}" \
  --exclude "*.pth" \
  --exclude "*optim*" \
  --exclude "*.onnx"

verify_paths "${DST}" \
  "model_index.json" \
  "transformer" \
  "vae"
