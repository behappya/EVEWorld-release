#!/usr/bin/env bash
# Cosmos-Predict2.5-2B (NVIDIA, 2B world model; Video2World maps a single image to video).
# GATED: accept the NVIDIA Open Model License on the model page and run hf auth login (or pass HF_TOKEN).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${HERE}/_common.sh"

REPO_ID="${REPO_ID:-nvidia/Cosmos-Predict2.5-2B}"
DST="${DST:-${XMODEL_ROOT}/cosmos_predict25_2b}"

activate_env
check_login_if_gated 1

echo "==== [Cosmos-Predict2.5-2B] starting download (gated) ===="
# Cosmos is not standard diffusers: pull the whole repo and prune only once inference works.
dl_repo "${REPO_ID}" "${DST}" \
  --exclude "*optim*"

echo "[dl] directory contents:"
ls -la "${DST}" 2>/dev/null | head -30
du -sh "${DST}" 2>/dev/null || true
if [[ -n "$(ls -A "${DST}" 2>/dev/null)" ]]; then
  echo "[dl] Cosmos-Predict2.5-2B directory is non-empty; check the listing above to confirm weights are complete."
else
  echo "[dl] directory is empty, likely a 401 from missing login. Check the log." >&2
  exit 1
fi
