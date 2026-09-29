#!/usr/bin/env bash
#SBATCH --job-name=xmodel_dreamgen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -euo pipefail

# Cross-model DreamGen batch I2V inference payload (runs inside a GPU kjob pod).
# Overrides are passed as KEY=VALUE.

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]]; then
  echo "Refusing to run outside a submitted job (no GPU). Submit via launch_xmodel_dreamgen_infer_kjob.sh" >&2
  exit 2
fi

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg} (pass overrides as KEY=VALUE)" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
INFER_DIR="${INFER_DIR:-${EVEWORLD_ROOT}/benchmarks/baselines/xmodel_infer}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_world1}"   # diffusers 0.39.0; all four I2V pipelines available
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_FAMILY="${MODEL_FAMILY:?Set MODEL_FAMILY=wan|wan_ti2v|cogvideox|cosmos}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH=/data/.../xmodels/<model>}"
XMODEL_EVAL_ROOT="${XMODEL_EVAL_ROOT:-${GAGI_ROOT:-$HOME/gagi}/gr1_dreamgen_eval/xmodel_eval}"
DATA_PATH="${DATA_PATH:-${GAGI_ROOT:-$HOME/gagi}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
RUN_NAME="${RUN_NAME:-${MODEL_FAMILY}_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${XMODEL_EVAL_ROOT}/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/run.log}"
SUMMARY_PATH="${SUMMARY_PATH:-${SAVE_DIR}/generation_summary.json}"

# Generation parameters (duration tiers: 93=5.8s / 157=9.8s / 253=15.8s @16fps)
NUM_FRAMES="${NUM_FRAMES:-93}"
FPS="${FPS:-16}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-768}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
SEED="${SEED:-6666}"
DATA_LIMIT="${DATA_LIMIT:-0}"          # set to 4 for smoke runs
DTYPE="${DTYPE:-bf16}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# CUDA_VISIBLE_DEVICES is unset on purpose: if the outer environment forwards it as a single card,
# the other 7 are masked and spawned processes cannot bind cuda:1-7. Restrict via GPU_IDS instead.
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
unset CUDA_VISIBLE_DEVICES

export HF_HOME="${HF_HOME:-${GAGI_ROOT:-$HOME/gagi}/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${SAVE_DIR}"
touch "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${INFER_DIR}"

echo "============================================"
echo "xmodel DreamGen infer"
echo "  host        = $(hostname)"
echo "  family      = ${MODEL_FAMILY}"
echo "  model_path  = ${MODEL_PATH}"
echo "  data_path   = ${DATA_PATH}"
echo "  save_dir    = ${SAVE_DIR}"
echo "  frames/fps  = ${NUM_FRAMES}/${FPS}   size=${WIDTH}x${HEIGHT}"
echo "  steps/cfg   = ${NUM_INFERENCE_STEPS}/${GUIDANCE_SCALE}  seed=${SEED} dtype=${DTYPE}"
echo "  data_limit  = ${DATA_LIMIT}"
echo "  gpu_ids     = ${GPU_IDS}"
echo "============================================"
nvidia-smi -L 2>&1 || true

"${PYTHON_BIN}" xmodel_dreamgen_infer.py \
  --model-family "${MODEL_FAMILY}" \
  --model-path "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --save-dir "${SAVE_DIR}" \
  --gpu-ids "${GPU_IDS}" \
  --num-frames "${NUM_FRAMES}" \
  --fps "${FPS}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --seed "${SEED}" \
  --data-limit "${DATA_LIMIT}" \
  --dtype "${DTYPE}" \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  --summary-path "${SUMMARY_PATH}"

echo "Inference done. Outputs in ${SAVE_DIR}"
