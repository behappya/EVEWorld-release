#!/usr/bin/env bash
#SBATCH --job-name=eve_lam_pretrain
#SBATCH --gpus-per-task=nvidia.com/gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# EVE LAD pretrain kjob payload (single GPU is enough: latents are offline, LAD has only ~0.4M params).
# Two steps: (1) if the latent cache is missing, offline-encode with the Wan VAE; (2) self-supervised LAD training + go/no-go check.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]]; then
  echo "Refusing to run outside a submitted job (no GPU). Submit via launch_eve_lam_kjob.sh" >&2
  exit 2
fi

for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-$HOME/gagi/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
VIDEO_DIR="${LAM_VIDEO_DIR:-${GAGI}/gr1_finetune_data/raw_hf/gr1}"
VAE_PATH="${VAE_PATH:-${GAGI}/giga_world_0_video_pretrain/vae}"
LAT_CACHE="${LAT_CACHE:-${GAGI}/eve_outputs/latents/gr1_real.pt}"
LAM_OUT="${LAM_OUT:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"

NUM_FRAMES="${NUM_FRAMES:-49}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-768}"
STEPS="${STEPS:-8000}"; WINDOW="${WINDOW:-12}"; BATCH="${BATCH:-8}"

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/eve:${PYTHONPATH:-}"
cd "${REPO_DIR}"

echo "=========================================="
echo " EVE LAM pretrain kjob"
echo "  video_dir  = ${VIDEO_DIR}"
echo "  lat_cache  = ${LAT_CACHE}"
echo "  lam_out    = ${LAM_OUT}"
echo "  frames=${NUM_FRAMES} hw=${HEIGHT}x${WIDTH} steps=${STEPS} window=${WINDOW} batch=${BATCH}"
echo "  python     = ${TRAIN_PYTHON}"
echo "=========================================="

# (1) encode latents. Skip when: cache exists, clip count >= MIN_CACHE_N, and not forced.
#     Guards against reusing a small smoke cache (e.g. 8 clips) and under-training.
MIN_CACHE_N="${MIN_CACHE_N:-50}"
need_encode=1
if [[ -f "${LAT_CACHE}" && "${FORCE_ENCODE:-0}" != "1" ]]; then
  cache_n=$("${TRAIN_PYTHON}" -c "import torch;print(len(torch.load('${LAT_CACHE}',map_location='cpu')['latents']))" 2>/dev/null || echo 0)
  if [[ "${cache_n}" -ge "${MIN_CACHE_N}" ]]; then
    echo "[lam] latent cache exists and is sufficient (${cache_n} >= ${MIN_CACHE_N}); skipping encode: ${LAT_CACHE}"
    need_encode=0
  else
    echo "[lam] latent cache has only ${cache_n} clips (< ${MIN_CACHE_N}); treated as smoke cache -> re-encoding full set"
  fi
fi
if [[ "${need_encode}" == "1" ]]; then
  echo "[lam] encoding latents -> ${LAT_CACHE}"
  "${TRAIN_PYTHON}" eveworld/method/scripts/encode_latents.py \
    --video-dir "${VIDEO_DIR}" --out "${LAT_CACHE}" \
    --vae-path "${VAE_PATH}" --num-frames "${NUM_FRAMES}" \
    --height "${HEIGHT}" --width "${WIDTH}" ${ENCODE_LIMIT:+--limit ${ENCODE_LIMIT}}
fi

# (2) train LAD + go/no-go
echo "[lam] training LAD + go/no-go validation"
"${TRAIN_PYTHON}" eveworld/method/scripts/train_lam.py \
  --latents "${LAT_CACHE}" --out "${LAM_OUT}" \
  --steps "${STEPS}" --window "${WINDOW}" --batch "${BATCH}"

echo "[lam] DONE. checkpoint: ${LAM_OUT}"
