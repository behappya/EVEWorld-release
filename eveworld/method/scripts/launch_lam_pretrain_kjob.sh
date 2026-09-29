#!/usr/bin/env bash
set -euo pipefail
# EVE LAD self-supervised pretrain (cluster kjob, single GPU).
# Two steps: offline-encode GR1 latents with the Wan VAE -> self-supervised LAD training
# + go/no-go (transition_error separates laziness from real motion).
# Outputs: ${GAGI}/eve_outputs/latents/gr1_real.pt (latent cache)
#       ${GAGI}/eve_outputs/lam/lam_gr1.pt      (LAD checkpoint)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = third_party/giga-world-0

export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_eve_lam_pretrain.sh}"

PASS=(
  "REPO_DIR=${REPO_DIR}"
  "GAGI_ROOT=${GAGI_ROOT:-$HOME/gagi}"
  "TRAIN_VENV=${TRAIN_VENV:-$HOME/gagi/envs/giga_world_train_venv}"
  "LAM_VIDEO_DIR=${LAM_VIDEO_DIR:-$HOME/gagi/gr1_finetune_data/raw_hf/gr1}"
  "LAT_CACHE=${LAT_CACHE:-$HOME/gagi/eve_outputs/latents/gr1_real.pt}"
  "LAM_OUT=${LAM_OUT:-$HOME/gagi/eve_outputs/lam/lam_gr1.pt}"
  "NUM_FRAMES=${NUM_FRAMES:-49}"
  "HEIGHT=${HEIGHT:-480}"
  "WIDTH=${WIDTH:-768}"
  "STEPS=${STEPS:-8000}"
  "WINDOW=${WINDOW:-12}"
  "BATCH=${BATCH:-8}"
)
[[ -n "${ENCODE_LIMIT:-}" ]] && PASS+=("ENCODE_LIMIT=${ENCODE_LIMIT}")
[[ -n "${FORCE_ENCODE:-}" ]] && PASS+=("FORCE_ENCODE=${FORCE_ENCODE}")

echo "[EVE] submitting LAD pretrain kjob (single GPU)"
echo "  JOB_SCRIPT = ${JOB_SCRIPT}"
echo "  steps=${STEPS:-8000} window=${WINDOW:-12} batch=${BATCH:-8} frames=${NUM_FRAMES:-49}"
echo "  latent cache -> ${LAT_CACHE:-.../eve_outputs/latents/gr1_real.pt}"
echo "  LAD ckpt   -> ${LAM_OUT:-.../eve_outputs/lam/lam_gr1.pt}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] would run:"
  echo "  ${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh \\"
  printf '    %s \\\n' "${PASS[@]}"
  exit 0
fi

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" "${PASS[@]}"
