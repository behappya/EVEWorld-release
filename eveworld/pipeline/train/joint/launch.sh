#!/usr/bin/env bash
# Joint training (L_id + augmentation) — 200-step full kjob submission entry
# point (stage 2 of arm 44).
# Pass-through rules as in eveworld/pipeline/tia/launch.sh. Prints the plan by default; only
# `submit` launches.
# Sentinels: grep 'aug-sentinel' (ret_proxy falling / l_rest flat) + grep
# 't4g-sentinel' (L_id falling / B gate normal).
# Any sentinel anomaly within the first 20 steps -> kill the job and investigate
# (probe discipline).

set -euo pipefail

REPO_DIR="third_party/giga-world-0"
PACKED="${GAGI_ROOT:-$HOME/gagi}/gr1_finetune_data/packed_data"
# Base model can be overridden by env (BASE_TRANSFORMER=<pretrain path> skips round0 and trains directly)
BASE_TRANSFORMER_PATH="${BASE_TRANSFORMER:-${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/anchor_models/round0_ema_st/transformer}"
ANNO_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/track4gen_probe/t4g_anno"
ASSETS_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/track4gen_probe/aug_assets"

# config/run can be overridden by env (Run A=joint, Run B=wmaponly ablation)
BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.pipeline.train.joint.config}"
OUT_ROOT="${OUT_ROOT:-${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/t4g_joint}"
RUN_NAME="${RUN_NAME:-t4g_joint200}"
MAX_STEPS="${MAX_STEPS:-200}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"

[[ -f "${PACKED}/config.json" ]] || { echo "missing packed"; exit 1; }
[[ -f "${BASE_TRANSFORMER_PATH}/diffusion_pytorch_model.safetensors" ]] || { echo "missing base model"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]] || { echo "missing idx2vid"; exit 1; }
N_ASSETS=$(ls "${ASSETS_DIR}"/*.npz 2>/dev/null | wc -l || echo 0)
[[ "${N_ASSETS}" -ge 60 ]] || { echo "aug_assets too few (${N_ASSETS})"; exit 1; }

TRAIN_PROJECT_DIR="${OUT_ROOT}/experiments"

run_submit() {
  cd "${REPO_DIR}"
  mkdir -p "${OUT_ROOT}"
  PACKED_DATA_DIR="${PACKED}" \
  OUTPUT_ROOT="${OUT_ROOT}" \
  TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR}" \
  RUN_NAME="${RUN_NAME}" \
  MAX_STEPS="${MAX_STEPS}" \
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
  BATCH_SIZE_PER_GPU=1 \
  GRADIENT_ACCUMULATION_STEPS=8 \
  GPU_IDS="0 1 2 3 4 5 6 7" \
    ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
      "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}" \
      "TRANSFORMER_MODEL_PATH=${BASE_TRANSFORMER_PATH}" \
      "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}" \
      "PACKED_DATA_DIR=${PACKED}" \
      "OUTPUT_ROOT=${OUT_ROOT}" \
      "TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}" \
      "RUN_NAME=${RUN_NAME}" \
      "MAX_STEPS=${MAX_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8"
}

case "${1:-}" in
  submit) run_submit ;;
  *) cat <<EOF
=========== Joint training L_id+aug · ${MAX_STEPS} steps (not submitted) ===========
  config   = ${BASE_CONFIG_MODULE}
  base     = round0_ema
  assets   = ${N_ASSETS} files | anno = t4g_anno
  batch    = 8x1xGA8=64 | checkpoint every ${CHECKPOINT_INTERVAL} steps (4 slots to pick from)
  submit   : bash $(basename "$0") submit
=====================================================================
EOF
  ;;
esac
