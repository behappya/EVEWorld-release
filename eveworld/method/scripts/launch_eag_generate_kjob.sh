#!/usr/bin/env bash
set -euo pipefail
# EVE EAG sampling-based generation (cluster kjob, single GPU).
# Default one-shot paired submit: baseline (w=0) + EAG (w=0.03), same seed, for the "does EAG reduce laziness" comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = third_party/giga-world-0
export REPO_DIR
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_eve_eag_generate.sh}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
LAM="${LAM:-${GAGI}/eve_outputs/lam/lam_gr1.pt}"
DATA_PATH="${DATA_PATH:-${GAGI}/gr1_dreamgen_eval/giga_input/gr1_dreamgen_it2v.json}"
EAG_ROOT="${EAG_ROOT:-${GAGI}/eve_outputs/eag_eval}"
SEED="${SEED:-6666}"; NUM_FRAMES="${NUM_FRAMES:-93}"; LIMIT="${LIMIT:-0}"
PAIR="${PAIR:-1}"                       # 1 = paired submit baseline+EAG
EAG_WEIGHT="${EAG_WEIGHT:-0.03}"

submit_one() {
  local w="$1" tag="$2"
  local save="${EAG_ROOT}/${tag}_seed${SEED}_f${NUM_FRAMES}"
  echo "----- submitting ${tag} (eag_weight=${w}) -> ${save} -----"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[DRY_RUN] submit: EAG_WEIGHT=${w} SAVE_DIR=${save} LAM=${LAM} DATA_PATH=${DATA_PATH} NUM_FRAMES=${NUM_FRAMES} SEED=${SEED} LIMIT=${LIMIT}"
    return 0
  fi
  # Submit in background: the submit script ends with exec kjobctl, which replaces/blocks the process; a foreground serial submit would stall on the first one
  # and the second paired job would never submit (observed bug). Use () & to isolate each submit in its own background subshell.
  ( "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" \
      "REPO_DIR=${REPO_DIR}" "GAGI_ROOT=${GAGI}" \
      "DATA_PATH=${DATA_PATH}" "LAM=${LAM}" "SAVE_DIR=${save}" \
      "EAG_WEIGHT=${w}" "NUM_FRAMES=${NUM_FRAMES}" "SEED=${SEED}" "LIMIT=${LIMIT}" ) &
  sleep 8   # stagger the two kjobctl submits to avoid contention
}

if [[ ! -f "${LAM}" && "${DRY_RUN:-0}" != "1" ]]; then
  echo "[ERR] LAD checkpoint not found: ${LAM} (run launch_lam_pretrain_kjob.sh first)" >&2
  exit 1
fi

echo "[EVE] EAG generation. lam=${LAM} seed=${SEED} frames=${NUM_FRAMES} limit=${LIMIT} pair=${PAIR}"
if [[ "${PAIR}" == "1" ]]; then
  submit_one 0 baseline
  submit_one "${EAG_WEIGHT}" "eag_w${EAG_WEIGHT}"
else
  submit_one "${EAG_WEIGHT}" "eag_w${EAG_WEIGHT}"
fi
wait   # wait for all background submits (else an early launcher exit kills incomplete subshells)
echo "[EVE] all submitted. After generation run on <save>/generated_only: eveworld/evaluation/tea/ncm.py + eveworld/evaluation/tea/qwen_laziness.py"
