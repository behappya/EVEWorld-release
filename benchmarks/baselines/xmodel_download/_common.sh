#!/usr/bin/env bash
# Shared download helpers for the dl_*.sh scripts (CPU-only, safe to re-run).
# Weights land under ${GAGI_ROOT}/xmodels and never enter the repo.

set -euo pipefail

export HF_HOME="${HF_HOME:-${GAGI_ROOT:-$HOME/gagi}/.hf_home}"
# Off by default: hf_transfer is unstable in some environments.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
# hf_xet is not installed, but hub 1.23 still tries Xet and crashes on repos such as Cosmos
# (RuntimeError: Unable to parse string as hex hash value); plain HTTP works.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-giga_world1}"   # hf==1.23.0
XMODEL_ROOT="${XMODEL_ROOT:-${GAGI_ROOT:-$HOME/gagi}/xmodels}"

activate_env() {
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
  if ! command -v hf >/dev/null 2>&1; then
    echo "[FATAL] hf CLI not found (env=${CONDA_ENV})." >&2
    exit 1
  fi
}

# Gated repos: warn instead of exiting (Wan is ungated).
check_login_if_gated() {
  local gated="$1"
  if [[ "${gated}" == "1" ]]; then
    if ! hf auth whoami >/dev/null 2>&1; then
      cat >&2 <<EOF
[WARN] this model is a gated repo; first:
   1) open the model page in a browser and click "Agree/Access repository"
   2) run: hf auth login  (paste the token from https://huggingface.co/settings/tokens)
   or pass HF_TOKEN=hf_xxx to this script.
   Without login the download returns 401. Trying to continue anyway...
EOF
    fi
  fi
}

# dl_repo <repo_id> <dst_dir> [extra hf download args...]
# Each --include/--exclude takes a single glob, so repeat the flag; extra globs would fall through
# to the positional args and download the whole repo.
dl_repo() {
  local repo_id="$1"; shift
  local dst="$1"; shift
  mkdir -p "${dst}"
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local log="${dst}/download_${ts}.log"
  echo "[dl] repo   = ${repo_id}"
  echo "[dl] dst    = ${dst}"
  echo "[dl] log    = ${log}"
  local token_args=()
  if [[ -n "${HF_TOKEN:-}" ]]; then token_args=(--token "${HF_TOKEN}"); fi
  hf download "${repo_id}" \
    --local-dir "${dst}" \
    "${token_args[@]}" \
    "$@" \
    2>&1 | tee "${log}"
}

# Usage: verify_paths <base_dir> <path1> <path2> ...
verify_paths() {
  local base="$1"; shift
  local ok=1
  for p in "$@"; do
    if [[ -e "${base}/${p}" ]]; then
      echo "  OK  ${base}/${p}"
    else
      echo "  MISSING  ${base}/${p}"; ok=0
    fi
  done
  du -sh "${base}" 2>/dev/null || true
  if [[ "${ok}" == "1" ]]; then
    echo "[dl] verify OK: ${base}"
  else
    echo "[dl] missing paths; check the log and re-run this script (downloads resume)." >&2
    return 1
  fi
}
