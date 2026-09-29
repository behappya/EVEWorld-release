#!/usr/bin/env bash
#SBATCH --job-name=flowwam_lpips_epe
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -euo pipefail

for arg in "$@"; do [[ "$arg" == *=* ]] || { echo "bad arg: $arg" >&2; exit 2; }; export "$arg"; done
[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]] && exit 2

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
FLOWWAM_ROOT="${FLOWWAM_ROOT:-$HOME/FlowWAM}"
MANIFEST="${MANIFEST:-${GAGI_ROOT:-$HOME/gagi}/flowwam/heldout_r250_v1/manifest.json}"
FLOW_ROOT="${FLOW_ROOT:-${GAGI_ROOT:-$HOME/gagi}/flowwam/heldout_r250_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${GAGI_ROOT:-$HOME/gagi}/flowwam/heldout_r250_v1/lpips_flow_epe_v2}"
# EVEWorld carries the LPIPS package; RAFT is imported from FlowWAM via PYTHONPATH.
PYTHON="${PYTHON:-$HOME/miniconda/envs/EVEWorld/bin/python}"

source "${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-EVEWorld}"
cd "$REPO_DIR"
export PYTHONPATH="${REPO_DIR}:${FLOWWAM_ROOT}:${FLOWWAM_ROOT}/inference:${FLOWWAM_ROOT}/training:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

for path in "$MANIFEST" "$FLOW_ROOT/arm_control_final_robot_only" "$FLOW_ROOT/arm_eve_final_robot_only"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 1; }
done

exec "$PYTHON" "$REPO_DIR/benchmarks/robotwin_flowwam/flowwam_lpips_flow_epe.py" \
  --manifest "$MANIFEST" --flow-root "$FLOW_ROOT" \
  --output-dir "$OUTPUT_DIR" --num-shards 8 --lpips-samples 8
