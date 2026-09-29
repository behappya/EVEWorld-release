#!/bin/bash
# L0: 7.8s long-horizon baseline candidate pool: 92 prompts x 8 seeds x 125 frames from the old full-param EMA baseline (gr1 SFT, 7.8s@16fps). One 8-GPU node, ~2h.
# Usage: bash l0_longhorizon_pool.sh [ARM=pretrain]
set -euo pipefail
REPO=third_party/giga-world-0
PRETRAIN="${GAGI_ROOT:-$HOME/gagi}/giga_world_0_video_pretrain"
EMA_CKPT="${GAGI_ROOT:-$HOME/gagi}/giga_world_0_outputs/gr1_finetune/experiments_200_clean"
EMA_CKPT="${EMA_CKPT}/models/checkpoint_epoch_100_step_200/transformer_ema"
ARM="${ARM:-gr1_sft_ema}"
OUT_BASE="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/longpool_f125"

if [[ "$ARM" == "gr1_sft_ema" ]]; then
  # the bestofn chain hard-codes the MODEL_DIR/{transformer,text_encoder,vae} layout;
  # adapt the EMA ckpt with a dir of symlinks
  MODEL_DIR="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/anchor_models/gr1_fullft_ema"
  mkdir -p "$MODEL_DIR"
  [[ -e "$MODEL_DIR/transformer"   ]] || ln -s "$EMA_CKPT" "$MODEL_DIR/transformer"
  [[ -e "$MODEL_DIR/text_encoder" ]] || ln -s "$PRETRAIN/text_encoder" "$MODEL_DIR/text_encoder"
  [[ -e "$MODEL_DIR/vae"          ]] || ln -s "$PRETRAIN/vae" "$MODEL_DIR/vae"
  [[ -d "$EMA_CKPT" ]] || { echo "missing canonical baseline EMA: $EMA_CKPT"; exit 1; }
elif [[ "$ARM" == "pretrain" ]]; then
  MODEL_DIR="$PRETRAIN"
else
  echo "unknown ARM=$ARM (expected gr1_sft_ema | pretrain)"; exit 1
fi

OUT_ROOT="$OUT_BASE/$ARM"
mkdir -p "$OUT_ROOT"
echo "== submitting long-horizon baseline pool: arm=$ARM  frames=125(7.8s)  out=$OUT_ROOT =="
# Note: OUT_ROOT is not in submit_gigaworld0_kjob.sh's maybe_export whitelist; it must be
# appended to the payload directly as KEY=VALUE ("$@" channel), otherwise it falls back
# to the default old bestofn dir.
cd "$REPO"
export REPO_DIR="$REPO"
export RL_DIR="${RL_DIR:-$HOME/new_rl/rl}"
export JOB_SCRIPT="$REPO/eveworld/method/scripts/kjob_bestofn_8gpu.sh"
export MODEL_DIR   # pass through the whitelist
bash "$REPO/scripts/submit_gigaworld0_kjob.sh" \
  "OUT_ROOT=$OUT_ROOT" \
  "SEEDS=6666 1234 2025 777 42 314 2718 999" \
  "NUM_FRAMES=125" \
  "SKIP_EXISTING=1" \
  "EAG_WEIGHT=0" \
  "LIMIT=0"

echo ""
echo "artifacts: $OUT_ROOT/seed{S}_f125/generated_only/*.mp4 (92 per seed)"
echo "next: qwen_28 scoring -> long-bucket severity distribution + 7.8s consensus " \
     "pairing (same rules as the short bucket)"
echo "note: kjob uses a full 8-GPU node; run T0 on a different node if concurrent"
