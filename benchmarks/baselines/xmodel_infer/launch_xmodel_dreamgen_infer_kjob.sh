#!/usr/bin/env bash
set -euo pipefail

# Submit cross-model DreamGen I2V inference to a GPU kjob (single pod), e.g.
#   MODEL_FAMILY=wan_ti2v MODEL_PATH=.../xmodels/wan22_ti2v_5b \
#   RUN_NAME=wan22_ti2v_5b_5p8s_smoke DATA_LIMIT=4 bash launch_xmodel_dreamgen_infer_kjob.sh
# Duration tier -> NUM_FRAMES @16fps: 5.8s=93  9.8s=157  15.8s=253

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
RL_DIR="${RL_DIR:-$HOME/new_rl/rl}"
JOB_SCRIPT="${JOB_SCRIPT:-${SCRIPT_DIR}/kjob_xmodel_dreamgen_infer.sh}"

: "${MODEL_FAMILY:?Set MODEL_FAMILY=wan|wan_ti2v|cogvideox|cosmos}"
: "${MODEL_PATH:?Set MODEL_PATH=/data/.../xmodels/<model>}"

if [[ ! -d "${RL_DIR}" ]]; then echo "Missing RL_DIR: ${RL_DIR}" >&2; exit 1; fi
if [[ ! -f "${JOB_SCRIPT}" ]]; then echo "Missing JOB_SCRIPT: ${JOB_SCRIPT}" >&2; exit 1; fi
if [[ ! -d "${MODEL_PATH}" ]]; then echo "Missing MODEL_PATH: ${MODEL_PATH} (download the weights first)" >&2; exit 1; fi

SCRIPT_ARGS=("REPO_DIR=${REPO_DIR}")
maybe_export() { local n="$1"; if [[ -n "${!n:-}" ]]; then SCRIPT_ARGS+=("${n}=${!n}"); fi; }

for name in \
  INFER_DIR CONDA_SH CONDA_ENV PYTHON_BIN \
  MODEL_FAMILY MODEL_PATH XMODEL_EVAL_ROOT DATA_PATH \
  RUN_NAME SAVE_DIR LOG_FILE SUMMARY_PATH \
  NUM_FRAMES FPS HEIGHT WIDTH NUM_INFERENCE_STEPS GUIDANCE_SCALE SEED \
  DATA_LIMIT DTYPE NEGATIVE_PROMPT \
  GPU_IDS CUDA_VISIBLE_DEVICES HF_HOME ; do
  maybe_export "${name}"
done

echo "Submitting xmodel DreamGen infer kjob"
echo "  family:     ${MODEL_FAMILY}"
echo "  model_path: ${MODEL_PATH}"
echo "  job_script: ${JOB_SCRIPT}"
echo "  run_name:   ${RUN_NAME:-<auto>}"
echo "  data_limit: ${DATA_LIMIT:-0}   frames: ${NUM_FRAMES:-93}"

if command -v kjobctl >/dev/null 2>&1; then
  KJOB_CMD=(kjobctl)
elif command -v pixi >/dev/null 2>&1; then
  cd "${RL_DIR}"; KJOB_CMD=(pixi run kjobctl)
else
  echo "Neither kjobctl nor pixi available." >&2; exit 1
fi

exec "${KJOB_CMD[@]}" create slurm \
  --pod-template-label vllm-metrics=true \
  -- \
  --chdir "${REPO_DIR}" \
  "${JOB_SCRIPT}" \
  "${SCRIPT_ARGS[@]}" \
  "$@"
