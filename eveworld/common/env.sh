#!/usr/bin/env bash
# EVE shared environment variables. All scripts `source` this file.

export GAGI_ROOT="${GAGI_ROOT:-$HOME/gagi}"
export GW0_MODEL_DIR="${GW0_MODEL_DIR:-${GAGI_ROOT}/giga_world_0_video_pretrain}"
export GR1_DATA_ROOT="${GR1_DATA_ROOT:-${GAGI_ROOT}/gr1_finetune_data}"

# Evaluation videos (input to the P0 metrics): P0 walks every *.mp4 below it; override here or with --video-dir.
export EVE_VIDEO_ROOT="${EVE_VIDEO_ROOT:-${GAGI_ROOT}/giga_world_0_outputs}"

export EVE_OUT="${EVE_OUT:-${GAGI_ROOT}/eve_outputs}"
# Checkpoints/videos go to ${GAGI_ROOT}/giga_world_0_outputs/eve/...

export EVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EVE_P0="${EVE_DIR}/diagnosis"
export PYBIN="${PYBIN:-python3}"

mkdir -p "${EVE_OUT}" 2>/dev/null || true

eve_log() { echo -e "\033[1;36m[EVE]\033[0m $*"; }
eve_warn() { echo -e "\033[1;33m[EVE WARN]\033[0m $*"; }
eve_err() { echo -e "\033[1;31m[EVE ERR]\033[0m $*" >&2; }
