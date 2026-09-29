#!/usr/bin/env bash
# ICH training, arm A (single layer block16) - kjob submit entry (71/B1 G3, 50-step probe).
# Only difference from arm B (eveworld/pipeline/ich/ich_3L_launch.sh) = config (t4g_ich_blocks).
# Three-layer passthrough rules same as eveworld/pipeline/selfcase/launch.sh. Default prints the plan;
# `submit` actually launches.
#
# Sentinels:
#   grep 'ich-sentinel' : det_pos/det_neg separate (case/background M means), gate/wout_norm
#   grep 'aug-sentinel' : ret_proxy should drop from ~0.9+ (A1 main criterion), l_rest flat

set -euo pipefail

REPO_DIR="third_party/giga-world-0"
PACKED="${GAGI_ROOT:-$HOME/gagi}/gr1_finetune_data/packed_data"
OUT_ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/t4g_ich_1L"
ROUND0_EMA_TRANSFORMER="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/track4gen_probe/t4g_anno"
CASE_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/selfcase/case_bank_all"

BASE_CONFIG_MODULE="eveworld.pipeline.ich.ich_1L_config"
RUN_NAME="${RUN_NAME:-t4g_ich_1L_s50}"
MAX_STEPS="${MAX_STEPS:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-25}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"

# preflight
[[ -f "${PACKED}/config.json" ]]                                   || { echo "missing ${PACKED}/config.json"; exit 1; }
[[ -f "${ROUND0_EMA_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]] || { echo "missing pretrain base"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                          || { echo "missing idx2vid"; exit 1; }
N_CASES=$(ls "${CASE_DIR}"/*.npz 2>/dev/null | wc -l || echo 0)
[[ "${N_CASES}" -ge 30 ]] || { echo "case_bank_all has only ${N_CASES} cases (<30); run eveworld/pipeline/selfcase/case_merge.py first"; exit 1; }

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
      "TRANSFORMER_MODEL_PATH=${ROUND0_EMA_TRANSFORMER}" \
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
=========== ICH arm A (block16 single layer) 50-step training (not submitted) ===========
  config     = ${BASE_CONFIG_MODULE}
  base       = ${ROUND0_EMA_TRANSFORMER}
  case_bank  = ${CASE_DIR} (${N_CASES} cases)
  batch      = 8x1xGA8=64, steps=${MAX_STEPS}, checkpoint every ${CHECKPOINT_INTERVAL} steps
  submit     : bash $(basename "$0") submit
  sentinels  : grep -E 'ich-sentinel|aug-sentinel' ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log
=================================================================
EOF
  ;;
esac
