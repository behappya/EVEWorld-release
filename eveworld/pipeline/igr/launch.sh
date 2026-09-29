#!/usr/bin/env bash
# Synthetic-copy augmentation: 50-step probe kjob entry point (item 44, stage 1').
# Three-level passthrough rules, same as eveworld/pipeline/tia/launch.sh: BASE_CONFIG_MODULE/
# TRANSFORMER_MODEL_PATH/CHECKPOINT_TOTAL_LIMIT must go as trailing KEY=VALUE.
# Dry-run by default; `submit` actually submits.

set -euo pipefail

REPO_DIR="third_party/giga-world-0"
PACKED="${GAGI_ROOT:-$HOME/gagi}/gr1_finetune_data/packed_data"
OUT_ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/t4g_aug"
ROUND0_EMA_TRANSFORMER="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/track4gen_probe/t4g_anno"
ASSETS_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/track4gen_probe/aug_assets"

BASE_CONFIG_MODULE="eveworld.pipeline.igr.config"
RUN_NAME="${RUN_NAME:-t4g_aug_probe50}"
PROBE_STEPS="${PROBE_STEPS:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"

# preflight
[[ -f "${PACKED}/config.json" ]]                                   || { echo "missing ${PACKED}/config.json"; exit 1; }
[[ -f "${ROUND0_EMA_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]] || { echo "missing base model"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                          || { echo "missing idx2vid mapping"; exit 1; }
N_ASSETS=$(ls "${ASSETS_DIR}"/*.npz 2>/dev/null | wc -l || echo 0)
[[ "${N_ASSETS}" -ge 60 ]] || { echo "aug_assets has only ${N_ASSETS} files (<60); run eveworld/pipeline/igr/prep_kjob.sh first"; exit 1; }

TRAIN_PROJECT_DIR="${OUT_ROOT}/experiments"

run_submit() {
  cd "${REPO_DIR}"
  mkdir -p "${OUT_ROOT}"
  PACKED_DATA_DIR="${PACKED}" \
  OUTPUT_ROOT="${OUT_ROOT}" \
  TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR}" \
  RUN_NAME="${RUN_NAME}" \
  MAX_STEPS="${PROBE_STEPS}" \
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL}" \
  BATCH_SIZE_PER_GPU=1 \
  GRADIENT_ACCUMULATION_STEPS=8 \
  GPU_IDS="0 1 2 3 4 5 6 7" \
    ./benchmarks/dreamgenbench/launch_gr1_train_kjob.sh \
      "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}" \
      "TRANSFORMER_MODEL_PATH=${ROUND0_EMA_TRANSFORMER}" \
      "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}" \
      "PACKED_DATA_DIR=${PACKED}" \
      "OUTPUT_ROOT=${OUT_ROOT}" \
      "TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}" \
      "RUN_NAME=${RUN_NAME}" \
      "MAX_STEPS=${PROBE_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8"
}

case "${1:-}" in
  submit) run_submit ;;
  *) cat <<EOF
=========== synthetic-copy augmentation - 50-step probe (not submitted) ===========
  config     = ${BASE_CONFIG_MODULE}
  base       = ${ROUND0_EMA_TRANSFORMER}
  assets     = ${ASSETS_DIR} (${N_ASSETS} files)
  batch      = 8x1xGA8=64, steps=${PROBE_STEPS}
  submit     : bash $(basename "$0") submit
  sentinel   : grep 'aug-sentinel' ${TRAIN_PROJECT_DIR}/logs/train_*.log
=========================================================
EOF
  ;;
esac
