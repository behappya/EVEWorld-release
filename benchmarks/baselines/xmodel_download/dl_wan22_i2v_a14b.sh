#!/usr/bin/env bash
# Wan2.2-I2V-A14B (Alibaba, MoE, ~27B total / 14B active). Not gated, ~55-65GB.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

REPO_ID="${REPO_ID:-Wan-AI/Wan2.2-I2V-A14B-Diffusers}"
DST="${DST:-${XMODEL_ROOT}/wan22_i2v_a14b}"

activate_env
check_login_if_gated 0

echo "==== [Wan2.2-I2V-A14B] starting download (large, be patient) ===="
dl_repo "${REPO_ID}" "${DST}" \
  --exclude "*.pth" \
  --exclude "*optim*" \
  --exclude "*.onnx"

verify_paths "${DST}" \
  "model_index.json" \
  "transformer" \
  "vae"
