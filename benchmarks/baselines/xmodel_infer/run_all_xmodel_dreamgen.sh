#!/usr/bin/env bash
set -uo pipefail

# Sequential one-shot runner: downloaded baseline models x DreamGen x {5.8s, 9.8s, 15.8s}.
# One GPU kjob per (model, duration), waiting for generation_summary.json before the next.
# Cosmos is not downloaded, so it is excluded by default.
#
# Overrides: SMOKE=1 (4 clips each), DURS="58" (single duration),
#            MODELS="wan_ti2v cogvideox" (subset of families).
# Duration -> frames differs per model (CogVideoX needs 16k+1). See frames_for() below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/launch_xmodel_dreamgen_infer_kjob.sh"
EVAL_ROOT="${XMODEL_EVAL_ROOT:-${GAGI_ROOT:-$HOME/gagi}/gr1_dreamgen_eval/xmodel_eval}"

XMODELS_DIR="${XMODELS_DIR:-${GAGI_ROOT:-$HOME/gagi}/xmodels}"

# Model list: "family:weight_dir_name". Add "cosmos:cosmos_predict25_2b" once downloaded.
DEFAULT_MODELS=("wan_ti2v:wan22_ti2v_5b" "wan:wan22_i2v_a14b" "cogvideox:cogvideox15_5b_i2v")

if [[ -n "${MODELS:-}" ]]; then
  SEL=()
  for m in "${DEFAULT_MODELS[@]}"; do
    fam="${m%%:*}"
    for want in ${MODELS}; do [[ "${fam}" == "${want}" ]] && SEL+=("${m}"); done
  done
  MODEL_LIST=("${SEL[@]}")
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

DURS="${DURS:-58 98 158}"

SMOKE="${SMOKE:-0}"
DATA_LIMIT_VAL=$([[ "${SMOKE}" == "1" ]] && echo 4 || echo 0)
POLL_TIMEOUT="${POLL_TIMEOUT:-14400}"   # max seconds to wait for a single job (default 4h)
POLL_INTERVAL="${POLL_INTERVAL:-60}"

# Frames per duration for every family: 93/157/253 uniformly.
# These are also valid for CogVideoX: latent_frames=(nf-1)//4+1 must be even (patch_size_t=2);
# 93/157/253 -> 24/40/64, all even; 97/161/257 (16k+1) give odd latents and fail, so they must not be used.
frames_for() {
  local fam="$1" dur="$2"
  case "${dur}" in 58) echo 93;; 98) echo 157;; 158) echo 253;; esac
}
dur_label() { case "$1" in 58) echo 5p8s;; 98) echo 9p8s;; 158) echo 15p8s;; esac; }

echo "============================================================"
echo "Sequential run: xmodel DreamGen"
echo "  models: ${MODEL_LIST[*]}"
echo "  durations: ${DURS}   SMOKE=${SMOKE} (DATA_LIMIT=${DATA_LIMIT_VAL})"
echo "  output root: ${EVAL_ROOT}"
echo "============================================================"

total=0; done_ok=0; failed_list=()
for entry in "${MODEL_LIST[@]}"; do
  fam="${entry%%:*}"; wdir="${entry##*:}"
  mpath="${XMODELS_DIR}/${wdir}"
  if [[ ! -d "${mpath}" ]]; then
    echo "[skip] ${fam}: weight directory not found: ${mpath}"
    continue
  fi
  for dur in ${DURS}; do
    nf="$(frames_for "${fam}" "${dur}")"
    lbl="$(dur_label "${dur}")"
    tag=$([[ "${SMOKE}" == "1" ]] && echo "_smoke" || echo "")
    run_name="${wdir}_${lbl}${tag}"
    save_dir="${EVAL_ROOT}/${run_name}"
    summary="${save_dir}/generation_summary.json"
    total=$((total+1))

    echo
    echo "------------------------------------------------------------"
    echo "[$(date +%H:%M:%S)] (${total}) submitting ${fam} ${lbl} frames=${nf} -> ${run_name}"
    echo "------------------------------------------------------------"
    # Delete the old summary so a stale file cannot be mistaken for completion
    rm -f "${summary}"

    MODEL_FAMILY="${fam}" MODEL_PATH="${mpath}" \
    RUN_NAME="${run_name}" DATA_LIMIT="${DATA_LIMIT_VAL}" \
    NUM_FRAMES="${nf}" \
    bash "${LAUNCH}" || { echo "[warn] submit returned non-zero, still waiting for outputs"; }

    # Poll until completion (summary appears)
    echo "[wait] waiting for ${summary} to appear (poll every ${POLL_INTERVAL}s, timeout ${POLL_TIMEOUT}s)..."
    waited=0; ok=0
    while [[ ${waited} -lt ${POLL_TIMEOUT} ]]; do
      if [[ -f "${summary}" ]]; then ok=1; break; fi
      sleep "${POLL_INTERVAL}"; waited=$((waited+POLL_INTERVAL))
    done

    if [[ ${ok} -eq 1 ]]; then
      got=$(grep -oE '"ok"[: ]+[0-9]+' "${summary}" | grep -oE '[0-9]+' | head -1)
      tot=$(grep -oE '"total"[: ]+[0-9]+' "${summary}" | grep -oE '[0-9]+' | head -1)
      echo "[done] ${run_name}: ok=${got}/${tot}  ($(date +%H:%M:%S))"
      done_ok=$((done_ok+1))
    else
      echo "[TIMEOUT] ${run_name} timed out without a summary, skipping to the next one. Check ${save_dir}/run.log"
      failed_list+=("${run_name}")
    fi
  done
done

echo
echo "============================================================"
echo "All done: ${done_ok}/${total} completed"
[[ ${#failed_list[@]} -gt 0 ]] && printf "  not completed / timed out: %s\n" "${failed_list[@]}"
echo "Outputs under: ${EVAL_ROOT}/<model>_<duration>${tag}/"
echo "============================================================"
