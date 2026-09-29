#!/usr/bin/env bash
# ICH decisive run, arm A (single layer block16) + positive-class weighting - kjob submit
# entry (71/B1 G3 overtime).
# Background: the 50-step verdict showed det separation of only +0.01 (warmup ate the whole
#       run + positive class ~2-5% unbalanced); ret_proxy 1L=0.350/3L=0.371 did not beat
#       the A1 baseline 0.282.
# Changes vs eveworld/pipeline/ich/ich_1L_launch.sh:
#   MAX_STEPS=100 (warmup still 50 -> the last 50 steps at full lambda_det)
#   T4G_DET_POS_WEIGHT=25 (BCE positive-class weighting, via env; trainer env takes
#     precedence over config; effectiveness check = pos_weight=25.0 on the
#     [t4g-ich] line in run.log)
#   fresh OUT_ROOT/RUN_NAME directory, separate from old runs.
#
# Sentinels:
#   grep 'ich-sentinel' : step60+ det_pos/det_neg should separate (>0.05 preferred)
#   grep 'aug-sentinel' : ret_proxy against A1@100 steps

set -euo pipefail

REPO_DIR="third_party/giga-world-0"
PACKED="${GAGI_ROOT:-$HOME/gagi}/gr1_finetune_data/packed_data"
OUT_ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/t4g_ich_1L_pw"
ROUND0_EMA_TRANSFORMER="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/track4gen_probe/t4g_anno"
CASE_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/selfcase/case_bank_all"

BASE_CONFIG_MODULE="eveworld.pipeline.ich.ich_1L_config"
RUN_NAME="${RUN_NAME:-t4g_ich_1L_pw_s350}"
MAX_STEPS="${MAX_STEPS:-350}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
DET_POS_WEIGHT="${DET_POS_WEIGHT:-25}"
DET_WARMUP="${DET_WARMUP:-50}"

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
      "GRADIENT_ACCUMULATION_STEPS=8" \
      "T4G_DET_POS_WEIGHT=${DET_POS_WEIGHT}" \
      "T4G_DET_WARMUP=${DET_WARMUP}"
}

case "${1:-}" in
  submit) run_submit ;;
  *) cat <<EOF
====== ICH arm A + pos_weight, 350 steps (not submitted) ======
  config       = ${BASE_CONFIG_MODULE}
  base         = ${ROUND0_EMA_TRANSFORMER}
  case_bank    = ${CASE_DIR} (${N_CASES} cases)
  batch        = 8x1xGA8=64, steps=${MAX_STEPS}, checkpoint every ${CHECKPOINT_INTERVAL} steps
  pos_weight   = ${DET_POS_WEIGHT} (env T4G_DET_POS_WEIGHT, warmup=${DET_WARMUP})
  submit       : bash $(basename "$0") submit
  verify       : grep 'pos_weight' ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log  (expect =25.0)
  sentinels    : grep -E 'ich-sentinel|aug-sentinel' ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log
=============================================================
EOF
  ;;
esac
