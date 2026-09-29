#!/usr/bin/env bash
# EVE data curation. Some steps need network (HF downloads).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../common/env.sh"
DC="${HERE}/.."
OUT="${EVE_OUT}/data"; mkdir -p "${OUT}"

eve_log "data 1/4: held-out split (split the 92 clips into train/eval to remove overlap)"
${PYBIN} "${DC}/scripts/make_holdout.py" \
  --meta "${GR1_DATA_ROOT}/raw_hf/metadata.csv" \
  --out-dir "${OUT}/splits" --eval-frac 0.3 --seed 0

eve_log "data 2/4: fetch the official DreamGen Bench 126 split (needs network/HF token)"
eve_warn "  # TODO(net): confirm the source of the official DreamGen split list, then add the download command"
echo "  ref: https://arxiv.org/abs/2505.12705 (DreamGen), HF: nvidia/PhysicalAI-Robotics-GR00T-GR1"
echo "  expected output: ${OUT}/dreamgen126/{env29,object50,behavior47}/ lists"

eve_log "data 3/4: pull an unlabeled manipulation-video subset for LAM pretraining (OXE / BridgeData)"
eve_warn "  # TODO(net): pick the OXE subset and add the tfds/HF download. Target: ${OUT}/lam_pretrain_videos/"
echo "  suggestion: a few Open X-Embodiment manipulation subsets (self-supervised, no action labels needed)"

eve_log "data 4/4: cross-embodiment / process eval sets (Something-Something V2 etc., optional)"
eve_warn "  # TODO(net): SthSthV2 requires an access request; placeholder at ${OUT}/crossdomain/"
eve_log "data curation skeleton done. Actually run: held-out split. The rest are network TODOs."
