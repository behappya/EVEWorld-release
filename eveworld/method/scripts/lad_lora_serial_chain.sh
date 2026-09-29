#!/usr/bin/env bash
# EVE LAD-LoRA ablation [single-node serial] orchestrator.
# Submit W_LAD ablations in queue order; next job only after the current kjob pod disappears.
# Run in background: nohup bash eveworld/method/scripts/lad_lora_serial_chain.sh > /tmp/lad_chain.log 2>&1 &
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_DIR}"

WEIGHTS="${WEIGHTS:-0.1 0.0 0.2}"     # queue order: main setting -> control -> strong regularization
SEED="${SEED:-6666}"
POLL="${POLL:-120}"                    # poll interval (s)
LAUNCH="eveworld/method/scripts/launch_lad_lora_train_kjob.sh"

# Poll only the specific job this chain submitted, not jobs on other nodes (e.g. best-of-N).
# Job name captured from the submit output "job.batch/slurm-profile-slurm-XXXXX created".
job_alive() {  # $1 = job name (slurm-profile-slurm-XXXXX)
  timeout 40 kubectl get pods 2>/dev/null | grep -E "$1" | grep -qiE "Running|Pending|ContainerCreating" && return 0 || return 1
}

for w in ${WEIGHTS}; do
  echo "======================================================"
  echo "[chain] $(date '+%F %T') submitting W_LAD=${w} SEED=${SEED}"
  echo "======================================================"
  RUN_NAME="$(date +%Y%m%d_%H%M%S)_eve_lad_lora_w${w}_seed${SEED}"
  OUT="$(bash "${LAUNCH}" "W_LAD=${w}" "SEED=${SEED}" "RUN_NAME=${RUN_NAME}" 2>&1)"
  echo "${OUT}" | tail -3
  JOB="$(echo "${OUT}" | grep -oE "slurm-profile-slurm-[a-z0-9]+" | head -1)"
  if [[ -z "${JOB}" ]]; then
    echo "[chain] !! job name not captured; submit may have failed; aborting chain." >&2
    exit 1
  fi
  echo "[chain] submitted ${RUN_NAME} -> job ${JOB}, waiting for start..."
  sleep 60   # give the kjob pod time to start

  # wait for this specific job to finish (pod disappears)
  while job_alive "${JOB}"; do
    sleep "${POLL}"
  done
  RUN_LOG="${OUTPUT_ROOT:-$HOME/gagi/giga_world_0_outputs/eve}/${RUN_NAME}/run.log"
  echo "[chain] $(date '+%T') W_LAD=${w} (job ${JOB}) finished. run.log tail:"
  tail -4 "${RUN_LOG}" 2>/dev/null || echo "  (no run.log)"
  # verify completion marker
  if grep -q "Step\[150/150\]" "${RUN_LOG}" 2>/dev/null; then
    echo "[chain] W_LAD=${w} training complete (Step[150/150])"
  else
    echo "[chain] W_LAD=${w} no completion marker; may have failed mid-run; continuing (check ${RUN_LOG} manually)"
  fi
done

echo "[chain] $(date '+%F %T') all ${WEIGHTS} finished (serial)."
echo "[chain] next: for each run checkpoint run generate_eag.py --lora <ckpt> --eag-weight 0 -> Qwen independent judge scores laziness + quality."
