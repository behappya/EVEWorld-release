#!/usr/bin/env bash
set -uo pipefail
# EVE track-B best-of-N post-processing: run judge A (selection) + judge B (independent eval)
# Then the selection/eval separation analysis (avoids circularity). Generation runs first via launch_bestofn_generate_kjob.sh.
# PY=<python with openai+cv2> overrides the eval environment (default dreamgenbench_eval_venv)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
BON="${BON_ROOT:-${GAGI}/eve_outputs/bestofn}"
TQ="${TQ_ROOT:-${GAGI}/eve_outputs/tea_qwen}"
SEEDS="${SEEDS:-6666 1234 2025 777 42 314 2718 999}"
NUM_FRAMES="${NUM_FRAMES:-93}"
LIMIT="${LIMIT:-0}"
CONC="${CONCURRENCY:-64}"
PY="${PY:-$HOME/gagi/envs/dreamgenbench_eval_venv/bin/python}"
QWEN_BASE="${QWEN_BASE:-127.0.0.1}"

echo "[bon] seeds=[${SEEDS}] frames=${NUM_FRAMES} limit=${LIMIT} py=${PY}"

sel_csvs=(); eval_csvs=(); cand_dirs=(); present_seeds=()
for sd in ${SEEDS}; do
  gen="${BON}/seed${sd}_f${NUM_FRAMES}/generated_only"
  if [[ ! -d "${gen}" ]]; then echo "[skip] missing ${gen}"; continue; fi
  cand_dirs+=("${gen}"); present_seeds+=("${sd}")
  a_csv="${TQ}/bon_seed${sd}_laziness.csv"
  b_csv="${TQ}/bon_seed${sd}_B_laziness.csv"
  sel_csvs+=("${a_csv}"); eval_csvs+=("${b_csv}")
  # judge A (5 signatures, selection); resume is on by default, scored items skipped
  echo "----- seed=${sd} judge A -----"
  "${PY}" eveworld/evaluation/tea/qwen_laziness.py --video-dir "${gen}" --run-name "bon_seed${sd}" \
    --judge a --qwen-base "${QWEN_BASE}" --concurrency "${CONC}" --limit "${LIMIT}" \
    2>&1 | grep -E "mean_severity|errors|completed [0-9]+/[0-9]+$" | tail -3
  # judge B (independent: process-completeness decomposition + frame-sampling phase 0.5)
  echo "----- seed=${sd} judge B (independent) -----"
  "${PY}" eveworld/evaluation/tea/qwen_laziness.py --video-dir "${gen}" --run-name "bon_seed${sd}_B" \
    --judge b --frame-offset 0.5 --qwen-base "${QWEN_BASE}" --concurrency "${CONC}" --limit "${LIMIT}" \
    2>&1 | grep -E "mean_severity|errors|completed [0-9]+/[0-9]+$" | tail -3
done

echo ""
echo "[bon] === selection/eval separation analysis (N=${#cand_dirs[@]}) ==="
"${PY}" eveworld/method/scripts/best_of_n_select.py \
  --cand-dirs "${cand_dirs[@]}" \
  --sel-csvs  "${sel_csvs[@]}" \
  --eval-csvs "${eval_csvs[@]}" \
  --seeds "${present_seeds[@]}" --scaling \
  --out "${BON}/selection_separated_f${NUM_FRAMES}.json"

echo ""
echo "[bon] done. results: ${BON}/selection_separated_f${NUM_FRAMES}.json"
echo "  main paper numbers: reduction_B (drop confirmed by the independent judge) + scaling (N=2/4/8) + circularity_gap (de-circularization evidence)"
