#!/usr/bin/env bash
#SBATCH --job-name=selfcase_g1
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -uo pipefail
[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}"; done
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
TP="${TRAIN_PYTHON:-${GAGI_ROOT:-$HOME/gagi}/envs/giga_world_train_venv/bin/python}"
OUT_DIR="${OUT_DIR:-${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/selfcase/g1}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
source "$CONDA_SH"; conda activate "${CONDA_ENV:-gigaworld}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/third_party/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"
# 12 条病例量小: 单进程单卡即可, 不分片
CUDA_VISIBLE_DEVICES=0 "$TP" selfcase/g1.py --out-dir "$OUT_DIR"
echo KJOB_DONE
