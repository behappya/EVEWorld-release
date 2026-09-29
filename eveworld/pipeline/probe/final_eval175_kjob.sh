#!/usr/bin/env bash
#SBATCH --job-name=t4g_final_eval175
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# t4g_final per-step EMA: DreamGenBench 126-prompt generation + manifest audit.
# Frozen protocol copied from doc 67 §12.2 (seed004, 30 steps, 93f/16fps, 768x480, EAG0), weights=EMA
# (consistent with gr92 in this campaign). Generation worker/prepare reuse doc-65 standard pieces.
# STEPS takes a space-separated step list (e.g. "50 100 150 200"), split across nodes.
set -uo pipefail
[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]] && { echo "no GPU"; exit 2; }
for arg in "$@"; do [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }; done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
GIGA_MODELS_DIR="${GIGA_MODELS_DIR:-${EVEWORLD_ROOT}/third_party/giga-models}"
GAGI="${GAGI:-$HOME/gagi}"
TRAIN_VENV="${TRAIN_VENV:-${GAGI}/envs/giga_world_train_venv}"
PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

STEPS="${STEPS:?required: space-separated step list}"
SEED="${SEED:-004}"
MODELDIR_ROOT="${GAGI}/eve_v2_outputs/t4g_final/experiments/models"
OUTPUT_ROOT="${OUTPUT_ROOT:-${GAGI}/eve_v2_outputs/t4g_final/eval175_seed004_ema}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${OUTPUT_ROOT}_eval}"

source "${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"; conda activate "${CONDA_ENV:-EVEWorld}"
export PYTHONPATH="${EVEWORLD_ROOT}:${GIGA_MODELS_DIR}:${REPO_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${GAGI}/.hf_home}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1
export PYTORCH_NVML_BASED_CUDA_CHECK=1
cd "${REPO_DIR}"
mkdir -p "${OUTPUT_ROOT}" "${MANIFEST_ROOT}"

for step in ${STEPS}; do
  model="final_ema_s${step}"
  probe="${MODELDIR_ROOT}/final_s${step}_infer_modeldir"
  [[ -s "${probe}/transformer/config.json" ]] || { echo "missing modeldir: ${probe}" >&2; exit 1; }

  echo "[generate] model=${model} seed=${SEED} probe=${probe}"
  "${PYTHON}" eveworld/evaluation/eval175_seed_sharded_dispatch.py \
    --model "${model}" \
    --model-dir "${probe}" \
    --data-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${OUTPUT_ROOT}" \
    --chain-name "seed${SEED}_${model}" \
    --seeds "${SEED}" \
    --gpu-count 8 \
    --python "${PYTHON}" \
    --worker-script "${EVEWORLD_ROOT}/eveworld/evaluation/eval175_multiseed_worker.py" \
    --num-inference-steps 30 \
    --num-frames 93 \
    --fps 16 \
    --height 480 \
    --width 768 \
    --skip-existing

  echo "[audit] model=${model}"
  "${PYTHON}" eveworld/evaluation/eval175_multiseed_prepare.py \
    --generation-root "${OUTPUT_ROOT}/${model}" \
    --input-root "${GAGI}/gr1_dreamgen_eval/giga_input" \
    --output-root "${MANIFEST_ROOT}/${model}" \
    --model-name "${model}" \
    --seeds "${SEED}"
  "${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["ready"] is True and d["manifest_count"] == 126 and not d["errors"], d' \
    "${MANIFEST_ROOT}/${model}/prepare_report.json"
  echo "[ready] ${model}"
done
echo KJOB_DONE
