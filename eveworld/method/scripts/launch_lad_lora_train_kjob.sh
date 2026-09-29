#!/usr/bin/env bash
# EVE LAD-LoRA training-time anti-laziness finetune (plan27 §20 main line).
# Freeze the pretrained LAD; use its soft-top-k transition_error on the backbone x0 prediction (sigma-gated) as a regularizer distilled into LoRA.
# Reuses the GigaWorld kjob train payload (physlatent pattern); frozen backbone + LoRA fits comfortably on 8×H20.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = third_party/giga-world-0
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

# Also accepts positional KEY=VALUE args (equivalent to env vars); avoids the `bash launch W_LAD=0.2` pitfall where they are silently ignored.
for _arg in "$@"; do
  [[ "${_arg}" == *=* ]] && export "${_arg}"
done

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/benchmarks/dreamgenbench/kjob_train_gr1_finetune.sh}"
export TRAIN_VENV="${TRAIN_VENV:-$HOME/gagi/envs/giga_world_train_venv}"
export DATA_ROOT="${DATA_ROOT:-$HOME/gagi/gr1_finetune_data}"
export PACKED_DATA_DIR="${PACKED_DATA_DIR:-${DATA_ROOT}/packed_data}"
export MODEL_DIR="${MODEL_DIR:-$HOME/gagi/giga_world_0_video_pretrain}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/gagi/giga_world_0_outputs/eve}"

# Define W_LAD/SEED early: TRAIN_PROJECT_DIR uses them to isolate ablation dirs (otherwise w=0.1/0.0/0.2 share one
# project_dir -> resumes collide and ckpts overwrite each other; hit twice already).
W_LAD="${W_LAD:-0.1}"
SEED="${SEED:-6666}"
BASE_CONFIG_MODULE="${BASE_CONFIG_MODULE:-eveworld.method.configs.eve_lad_lora}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-gradbal_v2}"
LAD_LORA_LR="${LAD_LORA_LR:-4.315837287515549e-05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
RESUME="${RESUME:-0}"

export TRAIN_PROJECT_DIR="${TRAIN_PROJECT_DIR:-${OUTPUT_ROOT}/experiments_lad_lora_${EXPERIMENT_TAG}_w${W_LAD}_seed${SEED}}"
export GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
# max_steps=50 (~35 epochs): train only up to where the old run diverged (step35), checkpoint densely to pick the best.
# (the old run diverged at step35 without grad clip; this run sets max_grad_norm=1.0 in the config to check it holds.)
export MAX_STEPS="${MAX_STEPS:-50}"
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-1}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-5}"           # steps 5/10/../50 -> 10 checkpoints
export CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:--1}"    # keep all (default 5 deletes early healthy ckpts)
export SKIP_TRAIN_ENV_SETUP="${SKIP_TRAIN_ENV_SETUP:-1}"   # venv already provisioned; skip reinstall
RUN_NAME="${RUN_NAME:-${TIMESTAMP}_eve_lad_lora_w${W_LAD}_seed${SEED}}"
export RUN_NAME   # launch_gr1 has a default + whitelist, so export takes effect

if [[ -d "${TRAIN_PROJECT_DIR}" ]] && find "${TRAIN_PROJECT_DIR}" -mindepth 1 -print -quit | grep -q .; then
  if [[ "${ALLOW_EXISTING_PROJECT:-0}" != "1" ]]; then
    echo "Refusing to reuse non-empty TRAIN_PROJECT_DIR: ${TRAIN_PROJECT_DIR}" >&2
    echo "Use a new EXPERIMENT_TAG, or explicitly set ALLOW_EXISTING_PROJECT=1." >&2
    exit 2
  fi
fi

PASS_ARGS=(
  "BASE_CONFIG_MODULE=${BASE_CONFIG_MODULE}"
  "RUN_NAME=${RUN_NAME}"
  "SEED=${SEED}"
  "WITH_EMA=${WITH_EMA:-0}"
  "NUM_FRAMES=${NUM_FRAMES:-93}"
  "HEIGHT=${HEIGHT:-480}"
  "WIDTH=${WIDTH:-768}"
  "FPS=${FPS:-16}"
  "W_LAD=${W_LAD}"
  "EXPERIMENT_TAG=${EXPERIMENT_TAG}"
  "LAD_LORA_LR=${LAD_LORA_LR}"
  "MAX_GRAD_NORM=${MAX_GRAD_NORM}"
  "RESUME=${RESUME}"
  "CHECKPOINT_TOTAL_LIMIT=${CHECKPOINT_TOTAL_LIMIT}"
  "LAM_CKPT=${LAM_CKPT:-$HOME/gagi/eve_outputs/lam/lam_gr1.pt}"
  "LAD_TOPK=${LAD_TOPK:-3}"
  "LAD_TAU=${LAD_TAU:-0.5}"
  "LAD_SIGMA_MAX=${LAD_SIGMA_MAX:-0.0}"
  "LAD_BALANCE_MAX_SCALE=${LAD_BALANCE_MAX_SCALE:-1.0}"
)

echo "[EVE] submitting LAD-LoRA training config=${BASE_CONFIG_MODULE} steps=${MAX_STEPS} seed=${SEED} W_LAD=${W_LAD}"
echo "[EVE] RUN_NAME=${RUN_NAME}"
echo "[EVE] TRAIN_PROJECT_DIR=${TRAIN_PROJECT_DIR}"
echo "[EVE] lr=${LAD_LORA_LR} DeepSpeed-gradient-clipping=${MAX_GRAD_NORM} resume=${RESUME}"
echo "[EVE] prerequisite: lam_gr1.pt must be a go/no-go-PASS LAD; venv ready (SKIP_TRAIN_ENV_SETUP=1)"
echo "[EVE] run serially on one node: W_LAD=0.0 control first, then W_LAD=0.1 once stable."
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] pass-args: ${PASS_ARGS[*]}"
  exit 0
fi
# launch_gr1 ends with "$@" -> submit ends with "$@" -> kjob for-arg export; trailing args are forwarded end to end.
exec "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_train_kjob.sh" "${PASS_ARGS[@]}"
