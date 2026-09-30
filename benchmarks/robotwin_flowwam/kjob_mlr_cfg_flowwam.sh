#!/usr/bin/env bash
#SBATCH --job-name=cfg_flowwam_mlr
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
set -euo pipefail

for arg in "$@"; do [[ "$arg" == *=* ]] || { echo "bad arg: $arg" >&2; exit 2; }; export "$arg"; done
[[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]] && exit 2

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
DREAMGEN_DATA_ROOT="${DREAMGEN_DATA_ROOT:-${GAGI_ROOT:-$HOME/gagi}/gr1_dreamgen_eval/giga_input}"
CFG_ROOT="${CFG_ROOT:-${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/cfg_grid_seed004}"
FLOW_MANIFEST="${FLOW_MANIFEST:-${GAGI_ROOT:-$HOME/gagi}/flowwam/heldout_r250_v1/manifest.json}"
FLOW_ROOT="${FLOW_ROOT:-${GAGI_ROOT:-$HOME/gagi}/flowwam/heldout_r250_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/cfg_flowwam_mlr_v1}"
PYTHON="${PYTHON:-${GAGI_ROOT:-$HOME/gagi}/envs/giga_world_train_venv/bin/python}"

source "${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-gigaworld}"
cd "$REPO_DIR"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/third_party/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

for path in "$DREAMGEN_DATA_ROOT" "$CFG_ROOT" "$FLOW_MANIFEST" "$FLOW_ROOT"; do
  [[ -e "$path" ]] || { echo "missing required path: $path" >&2; exit 1; }
done

exec "$PYTHON" benchmarks/worldarena/mlr_cfg_flowwam_dispatch.py \
  --dreamgen-data-root "$DREAMGEN_DATA_ROOT" \
  --cfg-root "$CFG_ROOT" \
  --flow-manifest "$FLOW_MANIFEST" \
  --flow-root "$FLOW_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --python "$PYTHON" \
  --num-shards 8 \
  --frame-count 24
