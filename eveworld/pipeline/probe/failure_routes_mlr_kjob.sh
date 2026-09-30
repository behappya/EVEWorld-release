#!/usr/bin/env bash
#SBATCH --job-name=failure_routes_mlr
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -euo pipefail

for arg in "$@"; do
  [[ "$arg" == *=* ]] || { echo "bad arg: $arg" >&2; exit 2; }
  export "$arg"
done

[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]] && exit 2

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
MANIFEST="${MANIFEST:?MANIFEST is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${GAGI_ROOT:-$HOME/gagi}/envs/giga_world_train_venv/bin/python}"

source "${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-gigaworld}"
export PYTHONPATH="${REPO_DIR}:${EVEWORLD_ROOT}/eveworld/pipeline:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${REPO_DIR}"

[[ -s "${MANIFEST}" ]] || { echo "missing manifest: ${MANIFEST}" >&2; exit 1; }
nvidia-smi
exec "${TRAIN_PYTHON}" eveworld/pipeline/probe/failure_routes_reval.py mlr-worker \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}" \
  --num-gpus 8 \
  --python "${TRAIN_PYTHON}" \
  --poll-seconds 30
