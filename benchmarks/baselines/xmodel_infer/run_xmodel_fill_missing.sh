#!/usr/bin/env bash
set -uo pipefail

# Cross-model DreamGen generation - fill in missing durations: serial jobs on one 8-GPU node,
# the next job starts after the previous one finishes (reuses launch_xmodel_dreamgen_infer_kjob.sh).
#
# As of 2026-07-14: 5.8s/9.8s/15.8s done (CogVideoX 15.8s unfinished); 3.8s and 7.8s are filled
# in by default (3 models x 2 durations = 6 jobs; INCLUDE_COG158=1 completes CogVideoX).
#
# Duration -> frames @16fps: 3.8s=61  5.8s=93  7.8s=125  9.8s=157  15.8s=253
#   (CogVideoX requires latent=(nf-1)/4+1 to be even: 61->16✓ 125->32✓ 253->64✓)
#
# Idempotent: skips a run that already has generation_summary.json with enough videos.
#
# Overrides: DURS="38" (single duration), INCLUDE_COG158=1 (also complete CogVideoX 15.8s),
#            MODELS="wan_ti2v cogvideox", SMOKE=1 (4 clips per duration), FORCE=1 (ignore existing outputs).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCH="${SCRIPT_DIR}/launch_xmodel_dreamgen_infer_kjob.sh"
EVAL_ROOT="${XMODEL_EVAL_ROOT:-${GAGI_ROOT:-$HOME/gagi}/gr1_dreamgen_eval/xmodel_eval}"
XMODELS_DIR="${XMODELS_DIR:-${GAGI_ROOT:-$HOME/gagi}/xmodels}"

# family:weight_dir_name
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

# Default durations to fill: 3.8s(38) + 7.8s(78).
DURS="${DURS:-38 78}"
INCLUDE_COG158="${INCLUDE_COG158:-0}"

SMOKE="${SMOKE:-0}"
FORCE="${FORCE:-0}"
DATA_LIMIT_VAL=$([[ "${SMOKE}" == "1" ]] && echo 4 || echo 0)
EXPECT_N=$([[ "${SMOKE}" == "1" ]] && echo 4 || echo 92)
POLL_TIMEOUT="${POLL_TIMEOUT:-14400}"
POLL_INTERVAL="${POLL_INTERVAL:-60}"

frames_for() { case "$1" in 38) echo 61;; 58) echo 93;; 78) echo 125;; 98) echo 157;; 158) echo 253;; *) echo 93;; esac; }
dur_label()  { case "$1" in 38) echo 3p8s;; 58) echo 5p8s;; 78) echo 7p8s;; 98) echo 9p8s;; 158) echo 15p8s;; esac; }

# Job list: "family:wdir:dur"
JOBS=()
for entry in "${MODEL_LIST[@]}"; do
  for dur in ${DURS}; do JOBS+=("${entry}:${dur}"); done
done
# Optional: complete CogVideoX 15.8s (currently 16/92)
if [[ "${INCLUDE_COG158}" == "1" ]]; then
  JOBS+=("cogvideox:cogvideox15_5b_i2v:158")
fi

echo "============================================================"
echo "Cross-model DreamGen fill-in generation (serial, 8 GPUs/node)"
echo "  models:   ${MODEL_LIST[*]}"
echo "  durations: ${DURS}$([[ "${INCLUDE_COG158}" == "1" ]] && echo " + cogvideox:158")"
echo "  SMOKE=${SMOKE} FORCE=${FORCE} expected clips=${EXPECT_N}"
echo "  output root: ${EVAL_ROOT}"
echo "  jobs to submit: ${#JOBS[@]}"
echo "============================================================"

total=0; done_ok=0; skipped=0; failed_list=()
for job in "${JOBS[@]}"; do
  IFS=':' read -r fam wdir dur <<< "${job}"
  mpath="${XMODELS_DIR}/${wdir}"
  lbl="$(dur_label "${dur}")"
  nf="$(frames_for "${dur}")"
  tag=$([[ "${SMOKE}" == "1" ]] && echo "_smoke" || echo "")
  run_name="${wdir}_${lbl}${tag}"
  save_dir="${EVAL_ROOT}/${run_name}"
  summary="${save_dir}/generation_summary.json"
  total=$((total+1))

  if [[ ! -d "${mpath}" ]]; then
    echo "[skip] ${run_name}: weight directory not found: ${mpath}"
    failed_list+=("${run_name}(no-weights)"); continue
  fi

  # Idempotent: skip if already complete
  if [[ "${FORCE}" != "1" && -f "${summary}" ]]; then
    have=$(ls "${save_dir}"/*.mp4 2>/dev/null | wc -l)
    if [[ "${have}" -ge "${EXPECT_N}" ]]; then
      echo "[have] ${run_name}: ${have} clips already exist, skipping (FORCE=1 to rerun)"
      skipped=$((skipped+1)); continue
    fi
  fi

  echo
  echo "------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] (${total}/${#JOBS[@]}) submitting ${fam} ${lbl} frames=${nf} -> ${run_name}"
  echo "------------------------------------------------------------"
  rm -f "${summary}"

  MODEL_FAMILY="${fam}" MODEL_PATH="${mpath}" \
  RUN_NAME="${run_name}" DATA_LIMIT="${DATA_LIMIT_VAL}" \
  NUM_FRAMES="${nf}" \
  bash "${LAUNCH}" || echo "[warn] submit returned non-zero, still waiting for outputs"

  echo "[wait] waiting for ${summary} (poll every ${POLL_INTERVAL}s, timeout ${POLL_TIMEOUT}s)..."
  waited=0; ok=0
  while [[ ${waited} -lt ${POLL_TIMEOUT} ]]; do
    [[ -f "${summary}" ]] && { ok=1; break; }
    sleep "${POLL_INTERVAL}"; waited=$((waited+POLL_INTERVAL))
    # Heartbeat: print one line after each poll so the foreground shows progress (clips generated + time waited)
    have=$(ls "${save_dir}"/*.mp4 2>/dev/null | wc -l)
    logmt=""
    [[ -f "${save_dir}/run.log" ]] && logmt=" | log updated $(stat -c '%y' "${save_dir}/run.log" 2>/dev/null | cut -d. -f1 | cut -d' ' -f2)"
    echo "  [$(date +%H:%M:%S)] ${run_name}: ${have}/${EXPECT_N} clips, waited $((waited/60))m${logmt}"
  done

  if [[ ${ok} -eq 1 ]]; then
    got=$(grep -oE '"ok"[: ]+[0-9]+' "${summary}" | grep -oE '[0-9]+' | head -1)
    tot=$(grep -oE '"total"[: ]+[0-9]+' "${summary}" | grep -oE '[0-9]+' | head -1)
    echo "[done] ${run_name}: ok=${got:-?}/${tot:-?}  ($(date +%H:%M:%S))"
    done_ok=$((done_ok+1))
  else
    echo "[TIMEOUT] ${run_name} timed out without a summary. Check ${save_dir}/run.log"
    failed_list+=("${run_name}")
  fi
done

echo
echo "============================================================"
echo "Finished: completed ${done_ok} / submitted $((total-skipped)) (skipped existing ${skipped})"
[[ ${#failed_list[@]} -gt 0 ]] && printf "  not completed: %s\n" "${failed_list[@]}"
echo "Outputs under: ${EVAL_ROOT}/<model>_<duration>/"
echo "  When done, run eveworld/evaluation/tea/qwen_laziness.py and eveworld/evaluation/tea/ncm.py on these directories (generated-only, no crop)"
echo "============================================================"
