#!/usr/bin/env bash
# CogVideoX1.5-5B-I2V (Zhipu, DiT 5B, native I2V). ~10-11GB.
# GATED: accept the CogVideoX License on the model page and run hf auth login (or pass HF_TOKEN).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

REPO_ID="${REPO_ID:-zai-org/CogVideoX1.5-5B-I2V}"
DST="${DST:-${XMODEL_ROOT}/cogvideox15_5b_i2v}"

activate_env
check_login_if_gated 1

echo "==== [CogVideoX1.5-5B-I2V] starting download (gated) ===="
dl_repo "${REPO_ID}" "${DST}" \
  --exclude "*.onnx"

verify_paths "${DST}" \
  "model_index.json" \
  "transformer" \
  "vae"
