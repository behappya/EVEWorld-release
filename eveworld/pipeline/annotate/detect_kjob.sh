#!/usr/bin/env bash
#SBATCH --job-name=t4g_detect
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# Per-frame GDINO detection over the 92 GT videos -> annotation cache (1 node, 8 GPU shards).
set -uo pipefail
if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]]; then
  echo "Refusing to run outside a submitted job." >&2; exit 2
fi
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}"; exit 1; }; done
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${GAGI_ROOT:-$HOME/gagi}/envs/giga_world_train_venv/bin/python}"
OUT_DIR="${OUT_DIR:-${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/track4gen_probe/t4g_anno}"
NGPU="${NGPU:-8}"
source "${CONDA_SH}"; conda activate "${CONDA_ENV:-EVEWorld}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/third_party/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"
nvidia-smi || true
python annotate/detect_dispatch.py "$NGPU" "$OUT_DIR" "$TRAIN_PYTHON"
echo "DETECT_DONE -> ${OUT_DIR}"
