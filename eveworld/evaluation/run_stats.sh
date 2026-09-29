#!/usr/bin/env bash
# EVE: stats aggregation (CPU-only). Bootstrap CI + paired tests + plots over the
# multi-model / multi-seed process metrics.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${HERE}/../../common/env.sh"
METRICS_DIR="${METRICS_DIR:-${EVE_OUT}/eval/metrics}"
OUT="${EVE_OUT}/eval/stats"; mkdir -p "${OUT}"
if [[ ! -d "${METRICS_DIR}" ]]; then
  eve_warn "no ${METRICS_DIR}; run the evaluation to generate metrics first."; exit 0
fi
${PYBIN} "${HERE}/aggregate_stats.py" --metrics-dir "${METRICS_DIR}" --out-dir "${OUT}"
eve_log "stats done -> ${OUT}"
