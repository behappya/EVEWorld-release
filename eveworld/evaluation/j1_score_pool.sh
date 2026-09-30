#!/bin/bash
# J1: batch scoring with both qwen judges (HTTP concurrency; no training GPUs); resumable, judge B primary + judge A for selection.
# Usage: bash j1_score_pool.sh longpool|round0
set -eu
source "${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-gigaworld}"
REPO=third_party/giga-world-0
QWEN_BASE="${QWEN_BASE:-127.0.0.1}"   # Qwen3.6-35B-A3B @ 8000, started 2026-07-18
POOL="${1:-longpool}"

case "$POOL" in
  longpool)
    ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/longpool_f125/gr1_sft_ema"
    TAG=longpool_f125 ; SUFFIX=f125 ;;
  round0)
    ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/pool_round0_f93"
    TAG=pool_round0_f93 ; SUFFIX=f93 ;;
  *) echo "unknown pool: $POOL (longpool|round0)"; exit 1 ;;
esac
OUT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/scores/$TAG"
mkdir -p "$OUT"

cd "$REPO"
for seed in 6666 1234 2025 777 42 314 2718 999; do
  VDIR="$ROOT/seed${seed}_${SUFFIX}/generated_only"
  [[ -d "$VDIR" ]] || { echo "skip (missing dir): $VDIR"; continue; }
  n=$(ls "$VDIR"/*.mp4 2>/dev/null | wc -l)
  echo "== seed $seed ($n videos) judge B =="
  python eveworld/evaluation/tea/qwen_laziness.py \
    --video-dir "$VDIR" --out-root "$OUT" --run-name "seed${seed}_B" \
    --qwen-base "$QWEN_BASE" --judge b --frame-offset 0.5 --concurrency 96
  echo "== seed $seed judge A =="
  python eveworld/evaluation/tea/qwen_laziness.py \
    --video-dir "$VDIR" --out-root "$OUT" --run-name "seed${seed}_A" \
    --qwen-base "$QWEN_BASE" --judge a --concurrency 96
done
echo ""
echo "all done: $OUT/seed*_{A,B}* ; next: stats + (round0) Gate-2 / (longpool) severity dist"