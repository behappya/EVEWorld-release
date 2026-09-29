#!/usr/bin/env bash
# EVE: cross-model generation + process metrics (cluster kjob).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${HERE}/../../common/env.sh"
eve_log "cross-model evaluation"
eve_warn "# TODO(cluster): generate held-out eval videos for each model (pretrain/sft/eve/public)"
cat <<TIP
  flow: 1) each model generates -> \${EVE_OUT}/eval/videos/<model>/
        2) run P0 event extraction + process metrics per model (reuse the p0_diagnosis
           scripts; CPU-only, can run on the login node)
        3) aggregate into \${EVE_OUT}/eval/metrics/<model>.json
        4) run_stats.sh for stats and plots
  public models: include any I2V/world model with available outputs, to show that
  laziness is a domain-wide phenomenon (section 4)
TIP
