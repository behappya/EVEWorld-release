#!/usr/bin/env bash
#SBATCH --job-name=t4g_ichD_gen
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# ICH-D pool generation: 92 prompts x 4 seeds x 93f (same protocol as pool_pretrain_f93).
# Differs from kjob_bestofn_8gpu.sh: --gen-script -> eveworld/pipeline/ich/ichD_generate.py (D-arm inference
# wrapper; bare transformer would skip ICH). transformer = D-arm step350 EMA slot.
set -uo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]]; then
  echo "Refusing to run outside a submitted job (no GPU)." >&2
  exit 2
fi
for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-${GAGI_ROOT:-$HOME/gagi}/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
CKPT_TRANSFORMER="${CKPT_TRANSFORMER:-${GAGI}/eve_v2_outputs/t4g_ich_D/experiments/models/checkpoint_epoch_175_step_350/transformer_ema}"
PRETRAIN="${PRETRAIN:-${GAGI}/giga_world_0_video_pretrain}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
OUT_ROOT="${OUT_ROOT:-${GAGI}/eve_v2_outputs/selfcase/pool_ichD_s350_f93}"
SEEDS="${SEEDS:-42 314 777 999}"
# GEN_SCRIPT: eveworld/pipeline/ich/ichD_generate.py (ICH on; gates via T4G_ICH_* env below) or
# generate_eag.py (bare backbone, ich_d.* dropped by from_pretrained = ICH off arm).
GEN_SCRIPT="${GEN_SCRIPT:-${EVEWORLD_ROOT}/eveworld/pipeline/ich/ichD_generate.py}"
export T4G_ICH_SIGMA_GATE="${T4G_ICH_SIGMA_GATE:-}"
export T4G_ICH_M_GATE="${T4G_ICH_M_GATE:-0}"

[[ -f "${CKPT_TRANSFORMER}/diffusion_pytorch_model.bin" ]] || { echo "missing D-arm EMA slot: ${CKPT_TRANSFORMER}"; exit 1; }

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/third_party/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export REPO_DIR
cd "${REPO_DIR}"

echo "== ICH-D pool gen: seeds=${SEEDS} ckpt=${CKPT_TRANSFORMER} -> ${OUT_ROOT} =="
echo "   gen_script=${GEN_SCRIPT} sigma_gate='${T4G_ICH_SIGMA_GATE}' m_gate=${T4G_ICH_M_GATE}"
nvidia-smi || true

"${TRAIN_PYTHON}" eveworld/method/scripts/bestofn_dispatch.py \
  --seeds ${SEEDS} \
  --data-path "${DATA_PATH}" --out-root "${OUT_ROOT}" \
  --transformer "${CKPT_TRANSFORMER}" \
  --text-encoder "${PRETRAIN}/text_encoder" \
  --vae "${PRETRAIN}/vae" \
  --lam "${LAM}" \
  --python "${TRAIN_PYTHON}" \
  --gen-script "${GEN_SCRIPT}" \
  --eag-weight 0 \
  --num-frames 93 --steps 30 --height 480 --width 768 --fps 16 \
  --limit "${LIMIT:-0}" --skip-existing
RC=$?
echo "KJOB_DONE rc=${RC}"
exit "${RC}"
