#!/usr/bin/env bash
#SBATCH --job-name=eve_bestofn_8gpu
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# EVE best-of-N generation -- one node, 8 GPUs in parallel (one GPU per seed, 92 clips serial per GPU).
# Replaces the old fragmented "one kjob per seed" setup (8 jobs on 7 nodes, one GPU each).
# 8 seeds dispatched in parallel to GPU 0..7 (Python dispatcher). Catch-up mode (--skip-existing) does not re-render existing mp4s.
set -uo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]]; then
  echo "Refusing to run outside a submitted job (no GPU)." >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gigaworld}"
TRAIN_VENV="${TRAIN_VENV:-$HOME/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
MODEL_DIR="${MODEL_DIR:-${GAGI}/giga_world_0_video_pretrain}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
OUT_ROOT="${OUT_ROOT:-${GAGI}/eve_outputs/bestofn}"

SEEDS="${SEEDS:-6666 1234 2025 777 42 314 2718 999}"
EAG_WEIGHT="${EAG_WEIGHT:-0}"          # 0 = baseline sampling (for best-of-N)
NUM_FRAMES="${NUM_FRAMES:-93}"
STEPS="${STEPS:-30}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-768}"; FPS="${FPS:-16}"
LIMIT="${LIMIT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"    # 1 = catch-up mode (default); 0 = regenerate everything

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/third_party/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

read -r -a SEED_ARR <<< "${SEEDS}"
echo "=========================================="
echo " EVE best-of-N single-node 8-GPU  host=$(hostname)"
echo "  seeds   = ${SEEDS}  (N=${#SEED_ARR[@]} -> GPU 0..$(( ${#SEED_ARR[@]} - 1 )), Python dispatcher)"
echo "  eag_w=${EAG_WEIGHT} frames=${NUM_FRAMES} steps=${STEPS} limit=${LIMIT} skip_existing=${SKIP_EXISTING}"
echo "  out     = ${OUT_ROOT}"
echo "=========================================="
nvidia-smi || true

SKIP_ARG=(); [[ "${SKIP_EXISTING}" == "1" ]] && SKIP_ARG=(--skip-existing)

# Multi-GPU orchestration lives in the Python dispatcher (kjobctl's slurm interpreter does not support bash background jobs).
"${TRAIN_PYTHON}" eveworld/method/scripts/bestofn_dispatch.py \
  --seeds ${SEEDS} \
  --data-path "${DATA_PATH}" --out-root "${OUT_ROOT}" \
  --transformer "${MODEL_DIR}/transformer" \
  --text-encoder "${MODEL_DIR}/text_encoder" \
  --vae "${MODEL_DIR}/vae" \
  --lam "${LAM}" \
  --python "${TRAIN_PYTHON}" \
  --gen-script "${EVEWORLD_ROOT}/eveworld/method/scripts/generate_eag.py" \
  --eag-weight "${EAG_WEIGHT}" \
  --num-frames "${NUM_FRAMES}" --steps "${STEPS}" \
  --height "${HEIGHT}" --width "${WIDTH}" --fps "${FPS}" \
  --limit "${LIMIT}" "${SKIP_ARG[@]}"
exit "$?"
