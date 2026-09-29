#!/usr/bin/env bash
# EVE causal-process-faithful full finetune (reuses the GigaWorld kjob payload, physlatent-style).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = third_party/giga-world-0
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh}"
export TRAIN_VENV="${TRAIN_VENV:-$HOME/gagi/envs/giga_world_train_venv}"
export DATA_ROOT="${DATA_ROOT:-$HOME/gagi/gr1_finetune_data}"
export PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
export MODEL_DIR="${MODEL_DIR:-$HOME/gagi/giga_world_0_video_pretrain}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/gagi/giga_world_0_outputs/eve}"
# Points at the EVE config module (runner=eveworld.EveCausalTrainer set in its config)
export BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.method.configs.eve_causal_fullft}"
export TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR:-${OUTPUT_ROOT}/experiments_causal_fullft}"
export RUN_NAME="${RUN_NAME:-${TIMESTAMP}_eve_causal_seed${SEED:-6666}}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-200}"
export NUM_FRAMES="${NUM_FRAMES:-93}"; export HEIGHT="${HEIGHT:-480}"
export WIDTH="${WIDTH:-768}"; export FPS="${FPS:-16}"; export WITH_EMA="${WITH_EMA:-1}"
export SEED="${SEED:-6666}"

echo "[EVE] submitting training config=${BASE_CONFIG_MODULE} steps=${MAX_STEPS} seed=${SEED}"
echo "[EVE] ablation usage examples:"
echo "  SEED=123 bash $0                                   # multiple seeds"
echo "  BASE_CONFIG_MODULE=eveworld.method.configs.eve_ablation_no_cpc bash $0"
# Reuses the existing submitter (invokes kjobctl on JOB_SCRIPT like physlatent does)
exec "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_train_kjob.sh"
