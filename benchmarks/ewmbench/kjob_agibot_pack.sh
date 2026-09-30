#!/usr/bin/env bash
#SBATCH --job-name=agibot_pack
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -uo pipefail
for arg in "$@"; do [[ "$arg" == *=* ]] && export "$arg"; done
GAGI="${GAGI_ROOT:-$HOME/gagi}"
VIDEO_DIR="${VIDEO_DIR:-${GAGI}/agibot_ewm_train_final}"
SAVE_DIR="${SAVE_DIR:-${GAGI}/agibot_ewm_packed}"
source "${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-gigaworld}"
cd third_party/giga-world-0
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "${SAVE_DIR}"
LOG="${GAGI}/agibot_ewm_pack.log"
python scripts/pack_data.py \
  --video-dir "${VIDEO_DIR}" \
  --save-dir "${SAVE_DIR}" \
  --text-encoder-model-path "${GAGI}/giga_world_0_video_pretrain/text_encoder" \
  > "${LOG}" 2>&1
rc=$?
echo "pack rc=${rc}" >> "${LOG}"
ls "${SAVE_DIR}" 2>/dev/null
exit "${rc}"
