#!/usr/bin/env bash
set -uo pipefail
# EVE track-B best-of-N: submit generation jobs for N different seeds (reuses generate_eag, plain w=0 by default).
# After generation, run eveworld/evaluation/tea/qwen_laziness.py on each seed dir, then select with best_of_n_select.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_eve_eag_generate.sh}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
BON_ROOT="${BON_ROOT:-${GAGI}/eve_outputs/bestofn}"
# Full scale: 8 seeds (N=8, supports N=2/4/8 scaling ablation) + LIMIT=0 (all 92 clips).
# LIMIT=16 is for smoke tests only; full runs must use LIMIT=0.
SEEDS="${SEEDS:-6666 1234 2025 777 42 314 2718 999}"
NUM_FRAMES="${NUM_FRAMES:-93}"
LIMIT="${LIMIT:-0}"
EAG_WEIGHT="${EAG_WEIGHT:-0}"       # 0 = plain generation candidates; >0 = candidates also use EAG (both tracks combined)

echo "[EVE] best-of-N generation. seeds=[${SEEDS}] frames=${NUM_FRAMES} limit=${LIMIT} eag_w=${EAG_WEIGHT}"
tag_wt=$([[ "${EAG_WEIGHT}" != "0" ]] && echo "_eag${EAG_WEIGHT}" || echo "")

for sd in ${SEEDS}; do
  save="${BON_ROOT}/seed${sd}${tag_wt}_f${NUM_FRAMES}"
  echo "----- submitting seed=${sd} -> ${save} -----"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] EAG_WEIGHT=${EAG_WEIGHT} SEED=${sd} SAVE_DIR=${save} NUM_FRAMES=${NUM_FRAMES} LIMIT=${LIMIT}"
    continue
  fi
  ( "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
      "REPO_DIR=${REPO_DIR}" "GAGI_ROOT=${GAGI}" \
      "DATA_PATH=${DATA_PATH}" "LAM=${LAM}" "SAVE_DIR=${save}" \
      "EAG_WEIGHT=${EAG_WEIGHT}" "NUM_FRAMES=${NUM_FRAMES}" "SEED=${sd}" "LIMIT=${LIMIT}" ) &
  sleep 8
done
wait
echo ""
echo "[EVE] all seeds submitted. After generation, score each seed with Qwen, then select:"
echo "  # 1) run Qwen on each seed dir:"
echo "  for sd in ${SEEDS}; do"
echo "    python eveworld/evaluation/tea/qwen_laziness.py --video-dir ${BON_ROOT}/seed\${sd}${tag_wt}_f${NUM_FRAMES}/generated_only \\"
echo "      --run-name bon_seed\${sd} --concurrency 64 --limit ${LIMIT}; done"
echo "  # 2) select (--cand-dirs and --qwen-csvs correspond one-to-one in seed order):"
echo "  python eveworld/method/scripts/best_of_n_select.py --cand-dirs <per-seed generated_only> \\"
echo "    --qwen-csvs <per-seed csv> --seeds ${SEEDS} --out ${BON_ROOT}/selection.json"
