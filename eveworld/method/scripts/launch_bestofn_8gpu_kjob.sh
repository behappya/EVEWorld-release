#!/usr/bin/env bash
# EVE best-of-N generation -- submit one [single-node 8-GPU] kjob (replaces the old fragmented 8 single-GPU jobs).
# 8 seeds fill the node's 8 GPUs in parallel, 92 clips serial per GPU; default catch-up mode (SKIP_EXISTING=1).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # = third_party/giga-world-0

export REPO_DIR
export RL_DIR="${RL_DIR:-${HOME}/new_rl/rl}"
export JOB_SCRIPT="${JOB_SCRIPT:-${EVEWORLD_ROOT}/eveworld/method/scripts/kjob_bestofn_8gpu.sh}"

# Overrides forwarded to the payload (submit_gigaworld0_kjob.sh maybe_exports these names)
export SEEDS="${SEEDS:-6666 1234 2025 777 42 314 2718 999}"
export NUM_FRAMES="${NUM_FRAMES:-93}"
export LIMIT="${LIMIT:-0}"
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export EAG_WEIGHT="${EAG_WEIGHT:-0}"
export SEED="${SEED:-6666}"   # placeholder; the payload dispatches on SEEDS

echo "[EVE] submitting best-of-N single-node 8-GPU job"
echo "  seeds=[${SEEDS}] frames=${NUM_FRAMES} limit=${LIMIT} skip_existing=${SKIP_EXISTING} eag_w=${EAG_WEIGHT}"
echo "  prerequisite: kill the old 8 single-GPU jobs first (else they hold GPUs twice)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] would submit ${JOB_SCRIPT} with SEEDS='${SEEDS}' LIMIT=${LIMIT} SKIP_EXISTING=${SKIP_EXISTING}"
  exit 0
fi

# submit_gigaworld0_kjob.sh needs SEEDS/SKIP_EXISTING/EAG_WEIGHT/OUT_ROOT on its maybe_export whitelist;
# anything not on the whitelist goes through EXTRA_ARGS as KEY=VALUE to the payload.
EXTRA_ARGS=("SEEDS=${SEEDS}" "SKIP_EXISTING=${SKIP_EXISTING}" "EAG_WEIGHT=${EAG_WEIGHT}"
            "NUM_FRAMES=${NUM_FRAMES}" "LIMIT=${LIMIT}")
exec "${REPO_DIR}/scripts/submit_gigaworld0_kjob.sh" "${EXTRA_ARGS[@]}"
