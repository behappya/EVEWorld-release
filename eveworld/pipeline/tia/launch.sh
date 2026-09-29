#!/usr/bin/env bash
# Track4Gen 式对应监督训练 —— 50 步探针 kjob 提交入口 (42 号 §3/§4)。
# kjobctl 不把工作机 env 带进 pod: 端到端只认尾随 KEY=VALUE 通道, env 仅给 launch 本地检查用。
# 监控: grep 't4g-sentinel' <train_log> —— 每步 [S1]四道loss裸值/nf [S2]param_disp_l2
#   [S3]sim 分布 [S4]探针纪律; 起步 grep '\[t4g\] block=', hook 用 blocks\[block17\] 行。

set -euo pipefail

REPO_DIR="third_party/giga-world-0"
GAGI="${GAGI_ROOT:-$HOME/gagi}"
DATA_ROOT="${GAGI}/gr1_finetune_data"
PACKED="${DATA_ROOT}/packed_data"                                   # 92 条 GT
OUT_ROOT="${GAGI}/eve_v2_outputs/t4g_corr"
ROUND0_EMA_TRANSFORMER="${GAGI}/eve_v2_outputs/anchor_models/round0_ema_st/transformer"
ANNO_DIR="${GAGI}/eve_v2_outputs/track4gen_probe/t4g_anno"

BASE_CONFIG_MODULE="eveworld.pipeline.tia.config"
RUN_NAME="${RUN_NAME:-t4g_corr_probe50}"
PROBE_STEPS="${PROBE_STEPS:-50}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-50}"
CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-8}"
CHECKPOINT_START_STEP="${CHECKPOINT_START_STEP:-0}"

# 预检 (缺件早失败, 不进 kjob)
[[ -f "${PACKED}/config.json" ]]                                   || { echo "缺 ${PACKED}/config.json"; exit 1; }
[[ -f "${ROUND0_EMA_TRANSFORMER}/diffusion_pytorch_model.safetensors" ]] || { echo "缺底座 ${ROUND0_EMA_TRANSFORMER}"; exit 1; }
[[ -f "${EVEWORLD_ROOT}/eveworld/pipeline/tia/config.py" ]]      || { echo "缺 eveworld/pipeline/tia/config.py"; exit 1; }
[[ -f "${ANNO_DIR}/_packidx2vid.json" ]]                          || { echo "缺 data_index->vid 映射 ${ANNO_DIR}/_packidx2vid.json"; exit 1; }
[[ -d "${ANNO_DIR}" ]]                                            || { echo "缺 anno 目录 ${ANNO_DIR}"; exit 1; }

TRAIN_PROJECT_DIR="${OUT_ROOT}/experiments"

# submit_cmd: 打印即事实; 用户执行 `submit` 才真正提交。
run_submit() {
  cd "${REPO_DIR}"
  mkdir -p "${OUT_ROOT}"
  # 前段走 env 通道 (launch 本地检查/banner); 尾随 KEY=VALUE 走 argv 通道 (pod 内最终生效)。
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
      "CHECKPOINT_START_STEP=${CHECKPOINT_START_STEP}" \
      "BATCH_SIZE_PER_GPU=1" \
      "GRADIENT_ACCUMULATION_STEPS=8"
}

print_plan() {
  cat <<EOF
================ Track4Gen 对应监督 · 50 步探针提交计划 (未提交) ================
  BASE_CONFIG_MODULE   = ${BASE_CONFIG_MODULE}
  TRANSFORMER (底座)   = ${ROUND0_EMA_TRANSFORMER}   (round0_ema, safetensors)
  PACKED_DATA (92 GT)  = ${PACKED}
  ANNO_DIR             = ${ANNO_DIR}
  OUTPUT_ROOT          = ${OUT_ROOT}
  RUN_NAME             = ${RUN_NAME}
  MAX_STEPS            = ${PROBE_STEPS}   (探针; gate 通过后再放全量)
  batch                = 8 卡 × 1/卡 × GA8 = 64
  CHECKPOINT           = interval ${CHECKPOINT_INTERVAL} / total_limit ${CHECKPOINT_TOTAL_LIMIT} / start_step ${CHECKPOINT_START_STEP}
--------------------------------------------------------------------------------
  真正提交:  bash $(basename "$0") submit
  监控:      tail -f ${OUT_ROOT}/kjob_logs/${RUN_NAME}.log
  哨兵:      grep 't4g-sentinel' ${TRAIN_PROJECT_DIR}/logs/train_*.log
================================================================================
EOF
}

case "${1:-}" in
  submit) run_submit ;;
  *)      print_plan ;;
esac
