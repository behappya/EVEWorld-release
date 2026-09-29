#!/usr/bin/env bash
#SBATCH --job-name=wmb_apre
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

# WMB adaptation A-pre training (node 56, T1): drive the existing runtime json
# (T4GJointTrainer) directly; the config is not regenerated. The 640-wide grid requires
# exporting T4G_W_LAT/T4G_WPIX (eveworld/pipeline/igr/paste.py compatibility patch).

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
GAGI="${GAGI_ROOT:-$HOME/gagi}"
TRAIN_VENV="${TRAIN_VENV:-${GAGI}/envs/giga_world_train_venv}"
RUNTIME_CONFIG="${RUNTIME_CONFIG:-${GAGI}/wmb_adapt/runtime_configs/t4g_wmb_apre_s600.json}"
LOG_DIR="${GAGI}/wmb_adapt/t4g_wmb_apre"
mkdir -p "${LOG_DIR}"
RUN_LOG="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
touch "${RUN_LOG}"
exec > >(tee -a "${RUN_LOG}") 2>&1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export HF_HOME="${GAGI}/.hf_home"
export T4G_W_LAT=40
export T4G_WPIX=640
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/third_party/giga-models:${PYTHONPATH:-}"
unset CUDA_VISIBLE_DEVICES

# shellcheck disable=SC1090
source "${TRAIN_VENV}/bin/activate"
TRAIN_PYTHON="${TRAIN_VENV}/bin/python"

# DeepSpeed: no nvcc on the node, so disable op building and CUDA detection (as in
# kjob_train_gr1_finetune.sh)
export DS_BUILD_OPS=0
export DS_BUILD_FP_QUANTIZER=0
export DS_IGNORE_CUDA_DETECTION=1
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda /usr/local/cuda-12.8 /usr/local/cuda-12 "${CONDA_PREFIX:-}/targets/x86_64-linux"; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
      export CUDA_HOME="${candidate}"
      break
    fi
  done
fi

cd "${REPO_DIR}"
echo "host=$(hostname) config=${RUNTIME_CONFIG} T4G_W_LAT=${T4G_W_LAT}"
nvidia-smi -L | head -2

# Fill in aug assets first if missing (idempotent: resume relies on npz overwrite), using the
# giga_world1 python (its transformers supports the GDINO chain)
N_AUG=$(ls "${GAGI}/wmb_adapt/aug_assets_v1/"*.npz 2>/dev/null | wc -l)
echo "aug assets present: ${N_AUG}/500"
if [[ "${N_AUG}" -lt "500" ]]; then
  N_SHARDS=16 "${GIGAWORLD1_PYTHON:-$HOME/miniconda/envs/giga_world1/bin/python}" \
    eveworld/agibot/w6_dispatch.py
  echo "aug fill rc=$? now=$(ls "${GAGI}/wmb_adapt/aug_assets_v1/"*.npz 2>/dev/null | wc -l)/500"
fi

"${TRAIN_PYTHON}" scripts/train.py --config "${RUNTIME_CONFIG}"
rc=$?
echo "train exit rc=${rc}"
exit "${rc}"
