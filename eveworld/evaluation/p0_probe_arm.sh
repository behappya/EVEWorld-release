#!/bin/bash
# P0-probe: quick generation + scoring for any checkpoint vs the Round-0 baseline.
# LIMIT=12 (default): directional check (first 12 prompts x 8 seeds, ~10min); LIMIT=0: full 92x8 for Gate-3.
# Usage: CKPT=/path/to/checkpoint_dir/transformer TAG=raft_s40 [LIMIT=12] bash p0_probe_arm.sh
set -euo pipefail
REPO=third_party/giga-world-0
PRETRAIN="${GAGI_ROOT:-$HOME/gagi}/giga_world_0_video_pretrain"
CKPT="${CKPT:?need CKPT=path to the transformer dir}"
TAG="${TAG:?need TAG=label}"
LIMIT="${LIMIT:-12}"
MODEL_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/anchor_models/probe_${TAG}"
OUT_ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/probe/${TAG}"

[[ -d "$CKPT" ]] || { echo "missing ckpt: $CKPT"; exit 1; }
mkdir -p "$MODEL_DIR" "$OUT_ROOT"
ln -sfn "$CKPT" "$MODEL_DIR/transformer"
ln -sfn "$PRETRAIN/text_encoder" "$MODEL_DIR/text_encoder"
ln -sfn "$PRETRAIN/vae" "$MODEL_DIR/vae"

echo "== submitting probe: $TAG (LIMIT=$LIMIT) =="
cd "$REPO"
export REPO_DIR="$REPO"
export RL_DIR="${RL_DIR:-$HOME/new_rl/rl}"
export JOB_SCRIPT="$REPO/eveworld/method/scripts/kjob_bestofn_8gpu.sh"
export MODEL_DIR
bash "$REPO/scripts/submit_gigaworld0_kjob.sh" \
  "OUT_ROOT=$OUT_ROOT" \
  "SEEDS=6666 1234 2025 777 42 314 2718 999" \
  "NUM_FRAMES=93" "SKIP_EXISTING=1" "EAG_WEIGHT=0" "LIMIT=$LIMIT"
echo "after generation, score + compare: bash p0_probe_score.sh $TAG"
