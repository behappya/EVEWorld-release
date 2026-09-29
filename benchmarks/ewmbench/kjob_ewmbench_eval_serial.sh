#!/usr/bin/env bash
#SBATCH --job-name=ewmbench_eval
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

set -uo pipefail

# Official EWMBench evaluation chain: resize -> YOLO trajectories -> evaluate (four
# dimensions), serial on one node; GT trajectories are detected once, on the first model
# (later runs pass --detect_gt to skip them). Captions go through config_eve.yaml (127.0.0.1:8000).

for arg in "$@"; do
  if [[ "${arg}" != *=* ]]; then
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
  export "${arg}"
done

EWM_DIR="${EWM_DIR:-${EWMBENCH_ROOT:-$HOME/gagibench/EWMBench}}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EWMBench}"
GAGI="${GAGI:-$HOME/gagi}"

MODELS="${MODELS:-pretrain}"
DIMENSIONS="${DIMENSIONS:-scene_consistency trajectory_consistency semantics diversity}"
# Passing OVERWRITE=1 on a second pass that only adds semantics recomputes this round's
# dimensions; with the incremental load in __init__.py only dimension_list entries are
# updated, so already computed scene/traj/diversity results survive.
OVERWRITE_FLAG=""
[[ -n "${OVERWRITE:-}" ]] && OVERWRITE_FLAG="--overwrite"
LAYOUT_ROOT="${LAYOUT_ROOT:-${GAGI}/eve_v2_outputs/ewmbench_gen/eval_layout}"
SAVE_ROOT="${SAVE_ROOT:-${GAGI}/eve_v2_outputs/ewmbench_eval}"
GT_DETECTED_FLAG="${SAVE_ROOT}/.gt_traj_done"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CAPTION_API_MODEL="${CAPTION_API_MODEL:-Qwen/Qwen3.6-35B-A3B}"

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
cd "${EWM_DIR}"

mkdir -p "${SAVE_ROOT}"
JOB_LOG="${SAVE_ROOT}/ewmbench_eval_$(date +%Y%m%d_%H%M%S).log"
touch "${JOB_LOG}"
echo "host=$(hostname) models=${MODELS} dims=${DIMENSIONS}" | tee -a "${JOB_LOG}"
nvidia-smi -L 2>&1 | head -2 | tee -a "${JOB_LOG}"

overall_rc=0
for model in ${MODELS}; do
  CFG="${SAVE_ROOT}/config_${model}.yaml"
  sed -e "s|^model_name: .*|model_name: ${model}|" \
      -e "s|DATANAME_dataset|${model}_dataset|" \
      -e "s|^save_path: .*|save_path: ${SAVE_ROOT}/${model}|" \
      config_eve.yaml > "${CFG}"
  mkdir -p "${SAVE_ROOT}/${model}"
  echo "===== $(date '+%F %T') START ${model} (config=${CFG})" | tee -a "${JOB_LOG}"

  echo "--- [${model}] video_resize" | tee -a "${JOB_LOG}"
  python processing/video_resize.py --config_path "${CFG}" >>"${SAVE_ROOT}/${model}/processing.log" 2>&1
  rc_a=$?

  # detection_tracking only feeds trajectory_consistency; skipped on semantics-only runs.
  if [[ "${DIMENSIONS}" == *trajectory* ]]; then
    echo "--- [${model}] detection_tracking" | tee -a "${JOB_LOG}"
    if [[ -f "${GT_DETECTED_FLAG}" ]]; then
      python processing/detection_tracking.py --config_path "${CFG}" --detect_gt >>"${SAVE_ROOT}/${model}/processing.log" 2>&1
      rc_b=$?
    else
      python processing/detection_tracking.py --config_path "${CFG}" >>"${SAVE_ROOT}/${model}/processing.log" 2>&1
      rc_b=$?
      if [[ "${rc_b}" == "0" ]]; then
        touch "${GT_DETECTED_FLAG}"
      fi
    fi
  else
    echo "--- [${model}] skip detection_tracking (no trajectory in DIMENSIONS)" | tee -a "${JOB_LOG}"
    rc_b=0
  fi

  echo "--- [${model}] evaluate: ${DIMENSIONS}" | tee -a "${JOB_LOG}"
  # shellcheck disable=SC2086
  python evaluate.py --dimension ${DIMENSIONS} --config_path "${CFG}" ${OVERWRITE_FLAG} >>"${SAVE_ROOT}/${model}/evaluate.log" 2>&1
  rc_c=$?

  echo "===== $(date '+%F %T') DONE ${model}: resize=${rc_a} detect=${rc_b} eval=${rc_c}" | tee -a "${JOB_LOG}"
  if [[ "${rc_a}${rc_b}${rc_c}" != "000" ]]; then
    overall_rc=1
  fi
done

echo "===== $(date '+%F %T') ALL DONE rc=${overall_rc}" | tee -a "${JOB_LOG}"
exit "${overall_rc}"
