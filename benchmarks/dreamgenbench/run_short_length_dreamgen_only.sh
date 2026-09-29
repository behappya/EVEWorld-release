#!/usr/bin/env bash
set -euo pipefail

# Run only the two DreamGen lines (Pretrain + SFT) at 3.8s / 7.8s, inference only; single
# 8-GPU node, serial and resumable. Equivalent to:
#   TASKS="dreamgen_pretrain dreamgen_sft" ./run_short_length_generation_only.sh
# Order: Pretrain 3.8s -> 7.8s (61/125 frames), then SFT 3.8s -> 7.8s. DRY_RUN=1 to preview.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS="dreamgen_pretrain dreamgen_sft" exec "${SCRIPT_DIR}/run_short_length_generation_only.sh" "$@"
