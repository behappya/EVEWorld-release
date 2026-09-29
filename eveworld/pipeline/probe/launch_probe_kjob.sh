#!/usr/bin/env bash
set -euo pipefail
# EVE · Track4Gen feature-traceability probe -- submit entry point (single node, 8 GPUs default).
# Goes through submit_gigaworld0_kjob.sh; custom vars passed to the payload as trailing KEY=VALUE
# (the payload receives them via `for arg in "$@"; do export`, see eveworld/pipeline/probe/probe_kjob.sh).
#
# usage:
#   bash eveworld/pipeline/probe/launch_probe_kjob.sh    # anmix_s200, 16 videos, 8 GPUs (defaults)
#   NGPU=4 bash ...                                    # 4 GPUs
#   VIDEO_IDS=13,32,76 SIGMAS=1.0 bash ...             # smoke (3 videos, 1 sigma)
#   MODEL_DIR=/data/.../probe_xxx OUT_DIR=/data/.../track4gen_probe/xxx bash ...
#   DRY_RUN=1 bash ...                                 # print only, no submit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = third_party/giga-world-0
export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/probe_kjob.sh}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
MODEL_DIR="${MODEL_DIR:-${GAGI}/eve_v2_outputs/anchor_models/probe_anmix_s200}"
VIDEO_ROOT="${VIDEO_ROOT:-${GAGI}/gr1_finetune_data/raw_data}"
VIDEO_IDS="${VIDEO_IDS:-13,32,76,14,15,16,17,18,19,20,21,23,24,25,26,27}"
SIGMAS="${SIGMAS:-0.25,0.7,2.0,5.0}"
NUM_FRAMES="${NUM_FRAMES:-93}"; HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-768}"; FPS="${FPS:-16}"
NGPU="${NGPU:-8}"; DTYPE="${DTYPE:-bf16}"
EPE_GO="${EPE_GO:-2.0}"; STAB_GO="${STAB_GO:-0.90}"; P_HI="${P_HI:-85}"; P_LO="${P_LO:-40}"
OUT_DIR="${OUT_DIR:-${GAGI}/eve_v2_outputs/track4gen_probe/anmix_s200}"

ARGS=(
  "REPO_DIR=${REPO_DIR}" "GAGI_ROOT=${GAGI}"
  "MODEL_DIR=${MODEL_DIR}"
  "VIDEO_ROOT=${VIDEO_ROOT}" "VIDEO_IDS=${VIDEO_IDS}"
  "SIGMAS=${SIGMAS}"
  "NUM_FRAMES=${NUM_FRAMES}" "HEIGHT=${HEIGHT}" "WIDTH=${WIDTH}" "FPS=${FPS}"
  "NGPU=${NGPU}" "DTYPE=${DTYPE}"
  "EPE_GO=${EPE_GO}" "STAB_GO=${STAB_GO}" "P_HI=${P_HI}" "P_LO=${P_LO}"
  "OUT_DIR=${OUT_DIR}"
)

echo "[t4g] submit Track4Gen probe"
echo "  model=${MODEL_DIR}"
echo "  videos=${VIDEO_IDS}"
echo "  sigmas=${SIGMAS}  ngpu=${NGPU}"
echo "  out=${OUT_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] ${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh ${ARGS[*]}"
  exit 0
fi

exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" "${ARGS[@]}"
