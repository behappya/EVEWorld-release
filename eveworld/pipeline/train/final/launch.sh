#!/usr/bin/env bash
# Final combined arm (arm 72) — kjob submission entry point.
# Production: 400 steps, checkpoint every 50, RUN_NAME=t4g_final_s400.
# smoke: `SMOKE=1 bash ... submit` -> 4 steps / every 2 / T4G_A2_P=0.5 (S1/S2/S3/S6),
#        RUN_NAME=t4g_final_smoke, separate OUT_ROOT subdir, leaves production runs alone.
# K final word = T4G_A2_K (filled in from the kprobe verdict).
#
# Sentinels:
#   grep 'a2-sentinel'  : sft/a2 mix, a2_hit, l_a2, m_sft/m_a2 (R1: sharp m drop = alarm)
#   grep 't4g-sentinel' : L_id raw value / effective frames (corr component)
#   grep 't4g-final'    : ICH-D load / hook installation

set -euo pipefail

REPO_DIR="third_party/giga-world-0"
PACKED="${GAGI_ROOT:-$HOME/gagi}/gr1_finetune_data/packed_data"
PRETRAIN_TRANSFORMER="${GAGI_ROOT:-$HOME/gagi}/giga_world_0_video_pretrain/transformer"
ANNO_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/track4gen_probe/t4g_anno"
ICH_D_WEIGHTS="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/selfcase/cic_match/ich_d_frozen_lr.npz"

BASE_CONFIG_MODULE="eveworld.pipeline.train.final.config"
SMOKE="${SMOKE:-0}"
if [[ "${SMOKE}" == "1" ]]; then
  OUT_ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/t4g_final/smoke"
  RUN_NAME="${RUN_NAME:-t4g_final_smoke}"
  MAX_STEPS="${MAX_STEPS:-4}"
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2}"
  A2_P="${A2_P:-0.5}"                 # smoke: raise the A2 share so both branches appear
else
  OUT_ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/t4g_final"
  RUN_NAME="${RUN_NAME:-t4g_final_s400}"
  MAX_STEPS="${MAX_STEPS:-400}"
  CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
  A2_P="${A2_P:-0.25}"
fi
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
A2_K="${A2_K:-4}"                     # kprobe verdict; must be confirmed before submitting

# preflight
[[ -f "${PACKED}/config.json" ]]                                          || { echo "missing ${PACKED}/config.json"; exit 1; }
[[ -f "${PRETRAIN_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]]    || { echo "missing pretrain base"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                                  || { echo "missing idx2vid"; exit 1; }
[[ -f "${ICH_D_WEIGHTS}" ]]                                               || { echo "missing frozen LR"; exit 1; }

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
      "TRANSFORMER_MODEL_PATH=${PRETRAIN_TRANSFORMER}" \
      "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}" \
      "PACKED_DATA_DIR=${PACKED}" \
      "OUTPUT_ROOT=${OUT_ROOT}" \
      "TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}" \
      "RUN_NAME=${RUN_NAME}" \
      "MAX_STEPS=${MAX_STEPS}" \
      "CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8" \
      "T4G_A2_K=${A2_K}" \
      "T4G_A2_P=${A2_P}"
}

case "${1:-}" in
  submit) run_submit ;;
  *) cat <<EOF
========== Final combined arm (not submitted, SMOKE=${SMOKE}) ==========
  config     = ${BASE_CONFIG_MODULE}
  base       = ${PRETRAIN_TRANSFORMER} (pretrain)
  components = SFT + L_id(0.5) + ICH-D frozen head + A2(p=${A2_P}, K=${A2_K})
  batch      = 8x1xGA8=64, steps=${MAX_STEPS}, checkpoint every ${CHECKPOINT_INTERVAL} steps
  submit     : [SMOKE=1] bash $(basename "$0") submit
  sentinels  : grep -E 'a2-sentinel|t4g-sentinel' ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log
=========================================================
EOF
  ;;
esac
