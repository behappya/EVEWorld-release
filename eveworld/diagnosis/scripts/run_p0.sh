#!/usr/bin/env bash
# EVE P0 go/no-go (CPU only, runs on any machine).
# Prereq: pip install -r eveworld/diagnosis/requirements.txt
#         and set EVE_VIDEO_ROOT / EVE_OUT in common/env.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../common/env.sh"

P0="${EVE_P0}"
OUT="${EVE_OUT}/p0"; mkdir -p "${OUT}"

# Per-model video dirs can come from the environment; defaults below (point them at your own artifacts)
BASELINE_DIR="${BASELINE_DIR:-${EVE_VIDEO_ROOT}/baseline}"
SFT_DIR="${SFT_DIR:-${EVE_VIDEO_ROOT}/sft}"
META="${GR1_META:-${GR1_DATA_ROOT}/raw_hf/metadata.csv}"

eve_log "P0 step 1/4: event extraction"
for tag in baseline sft; do
  d="BASELINE_DIR"; [[ "$tag" == "sft" ]] && d="SFT_DIR"
  vdir="${!d}"
  if [[ -d "$vdir" ]]; then
    eve_log "  extract $tag <- $vdir"
    ${PYBIN} "${P0}/scripts/event_extract.py" --video-dir "$vdir" --meta "$META" --out-dir "${OUT}/events_${tag}"
  else
    eve_warn "  skip $tag: directory does not exist $vdir (set ${d}=... to your generated videos)"
  fi
done

eve_log "P0 step 2/4: process metrics"
for tag in baseline sft; do
  [[ -d "${OUT}/events_${tag}" ]] || continue
  ${PYBIN} "${P0}/metrics/process_metrics.py" --events-dir "${OUT}/events_${tag}" \
    --out "${OUT}/metrics_${tag}.json" --tag "$tag"
done

eve_log "P0 step 3/4: orthogonality (needs your physics-score table PHYS_CSV, column PHYS_COL)"
PHYS_CSV="${PHYS_CSV:-}"; PHYS_COL="${PHYS_COL:-pbench_domain}"
if [[ -n "$PHYS_CSV" && -f "${OUT}/metrics_baseline.json" ]]; then
  ${PYBIN} "${P0}/metrics/orthogonality.py" --process "${OUT}/metrics_baseline.json" \
    --physics "$PHYS_CSV" --phys-col "$PHYS_COL" --out-prefix "${OUT}/ortho_baseline"
else
  eve_warn "  skip orthogonality: set PHYS_CSV=your per-video physics table (video,${PHYS_COL})"
fi

eve_log "P0 step 4/4: manual annotation prompt (go/no-go)"
cat <<TIP
  next (manual, required):
   1) sample 150-300 clips, one run per annotator:
      ${PYBIN} ${P0}/annotation/annotate.py --video-dir <SAMPLED_DIR> --out ${OUT}/lab_A.jsonl --resume
      ${PYBIN} ${P0}/annotation/annotate.py --video-dir <SAMPLED_DIR> --out ${OUT}/lab_B.jsonl --resume
   2) agreement κ:
      ${PYBIN} ${P0}/annotation/agreement.py kappa --a ${OUT}/lab_A.jsonl --b ${OUT}/lab_B.jsonl
   3) calibrate the auto metrics:
      ${PYBIN} ${P0}/annotation/agreement.py calib --labels ${OUT}/lab_A.jsonl --metrics ${OUT}/metrics_baseline.json

  go/no-go criteria (all three must pass before the method stage):
    frequency: LAZINESS_RATE high enough (see metrics_*.json)
    testability: calib agreement≥0.7 or spearman≥0.6
    orthogonality: ortho_baseline.json pearson |r|<0.4 and samples with high physics but cheating
TIP
eve_log "P0 automatic part done. Artifacts: ${OUT}"
