#!/usr/bin/env bash
#SBATCH --job-name=wmb_detect
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

# WMB adaptation: GDINO detection over the 500 trainset_v1 rows, sharded over 8 GPUs
# (orchestrated by the Python dispatcher).

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-gigaworld}"
ADAPT_DIR="${ADAPT_DIR:-eveworld/agibot}"
LOG_DIR="${GAGI_ROOT:-$HOME/gagi}/wmb_adapt"
mkdir -p "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset CUDA_VISIBLE_DEVICES

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${ADAPT_DIR}"

python w4_dispatch.py >"${LOG_DIR}/w4_detect.log" 2>&1
rc=$?
n=$(ls "${GAGI_ROOT:-$HOME/gagi}/wmb_adapt/t4g_anno/"*.json 2>/dev/null | wc -l)
echo "wmb detect done rc=${rc} anno=${n}/500" | tee -a "${LOG_DIR}/w4_detect.log"
exit "${rc}"
