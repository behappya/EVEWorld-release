#!/usr/bin/env bash
#SBATCH --job-name=selfcase_cic
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -uo pipefail
[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}"; done
REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
TP="${TRAIN_PYTHON:-${GAGI_ROOT:-$HOME/gagi}/envs/giga_world_train_venv/bin/python}"
OUT_DIR="${OUT_DIR:-${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/selfcase/cic_match_v2}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
source "$CONDA_SH"; conda activate "${CONDA_ENV:-EVEWorld}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/third_party/giga-models:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"
# 40 例 x 2 sigma, 单卡单进程串行即可 (整节点占位防碎片化, 与 eveworld/pipeline/selfcase/g1p_kjob.sh 同款)
CUDA_VISIBLE_DEVICES=0 "$TP" selfcase/cic_match.py --out-dir "$OUT_DIR"
echo KJOB_DONE
