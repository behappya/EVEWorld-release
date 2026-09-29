#!/usr/bin/env bash
set -euo pipefail

# Fill-in driver for short durations, inference only (no evaluation / judge / PA-II / Domain).
# Single 8-GPU node, serial and resumable: a duration with a complete generation_summary.json is
# skipped, one already running (run.log/submit.log present) is waited on instead of resubmitted.
# Frames at 16fps: 3.8s=61, 7.8s=125 (VAE 4k+1 and <= max_frames=128, so no temporal extrapolation).
# Overrides: TASKS=, LENGTHS=, DRY_RUN=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults match existing sweep outputs so results merge into the existing curves.
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE:-1}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-30}"
FPS="${FPS:-16}"
SEED="${SEED:-6666}"
DRY_RUN="${DRY_RUN:-0}"

# Durations to fill (label:frames, space separated).
LENGTHS="${LENGTHS:-3p8s:61 7p8s:125}"
# PBench only fills 7.8s (3.8s already exists).
PBENCH_LENGTHS="${PBENCH_LENGTHS:-7p8s:125}"

TASKS="${TASKS:-dreamgen_pretrain dreamgen_sft pbench_pretrain pbench_sft}"

EVAL_ROOT="${EVAL_ROOT:-${GAGI_ROOT:-$HOME/gagi}}"
PBENCH_DATA_PATH="${PBENCH_DATA_PATH:-${HOME}/gagibench/pbench/giga_input/pbench_robot_it2v.json}"
PBENCH_EXPECTED="${PBENCH_EXPECTED:-174}"
PBENCH_HEIGHT="${PBENCH_HEIGHT:-480}"
PBENCH_WIDTH="${PBENCH_WIDTH:-640}"
PBENCH_PRETRAIN_GEN_ROOT="${PBENCH_PRETRAIN_GEN_ROOT:-${EVAL_ROOT}/giga_world_0_outputs/pbench_robot_length_sweep}"
PBENCH_PRETRAIN_SWEEP_ID="${PBENCH_PRETRAIN_SWEEP_ID:-pbench_len_sweep_full_8gpu}"
PBENCH_SFT_GEN_ROOT="${PBENCH_SFT_GEN_ROOT:-${EVAL_ROOT}/giga_world_0_outputs/gr1_pbench_robot_length_sweep}"
PBENCH_SFT_SWEEP_ID="${PBENCH_SFT_SWEEP_ID:-gr1_pbench_len_sweep_full_8gpu}"

# SFT checkpoint (shared by PBench SFT and DreamGen SFT)
SFT_CHECKPOINT_DIR="${SFT_CHECKPOINT_DIR:-${EVAL_ROOT}/giga_world_0_outputs/gr1_finetune/experiments_200_clean/models/checkpoint_epoch_100_step_200}"
GR1_TRANSFORMER_SUBDIR="${GR1_TRANSFORMER_SUBDIR:-transformer_ema}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${EVAL_ROOT}/giga_world_0_video_pretrain}"
# Synthetic model directory for PBench SFT (transformer_ema + symlinks reusing pretrain's text_encoder/vae)
PBENCH_SFT_MODEL_DIR="${PBENCH_SFT_MODEL_DIR:-${PBENCH_SFT_GEN_ROOT}/model_gr1_checkpoint_epoch_100_step_200_${GR1_TRANSFORMER_SUBDIR}}"

DREAMGEN_EVAL_ROOT="${DREAMGEN_EVAL_ROOT:-${EVAL_ROOT}/gr1_dreamgen_eval}"
DREAMGEN_OUTPUT_ROOT="${DREAMGEN_OUTPUT_ROOT:-${DREAMGEN_EVAL_ROOT}/generated_side_by_side}"
DREAMGEN_EXPECTED="${DREAMGEN_EXPECTED:-92}"
DREAMGEN_HEIGHT="${DREAMGEN_HEIGHT:-480}"
DREAMGEN_WIDTH="${DREAMGEN_WIDTH:-768}"
# Fixed SWEEP_ID (not a timestamp) so run_name is deterministic and resumable
DREAMGEN_SWEEP_ID="${DREAMGEN_SWEEP_ID:-short_3p8_7p8}"

json_value() {
  local path="$1" key="$2"
  python3 - "$path" "$key" <<'PY'
import json, sys
path, key = sys.argv[1:3]
try:
    with open(path, "r", encoding="utf-8") as f:
        print(json.load(f).get(key, ""))
except Exception:
    print("")
PY
}

generation_done() {  # $1=save_dir  $2=expected
  local summary="$1/generation_summary.json"
  [[ -f "${summary}" ]] || return 1
  local gen req
  gen="$(json_value "${summary}" generated_count)"
  req="$(json_value "${summary}" request_count)"
  [[ "${gen}" == "$2" && "${req}" == "$2" ]]
}

mp4_count() { find "$1" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l; }

wait_for_generation() {  # $1=run_name  $2=save_dir  $3=expected
  local run_name="$1" save_dir="$2" expected="$3"
  local log_path="${save_dir}/run.log"
  echo "  waiting for generation: ${run_name}"
  while true; do
    if [[ -f "${log_path}" ]] && rg -q "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}"; then
      echo "  generation appears to have failed, check the log: ${log_path}" >&2
      rg -n "Traceback|AssertionError|ProcessRaisedException|CUDA out|Killed|RuntimeError" "${log_path}" -S | tail -n 60 >&2
      exit 1
    fi
    if generation_done "${save_dir}" "${expected}"; then
      echo "  generation complete: ${expected}/${expected}"
      break
    fi
    echo "    current mp4 count: $(mp4_count "${save_dir}")/${expected}; checking again in 60s"
    sleep 60
  done
}

prepare_sft_model_dir() {  # idempotent: build the synthetic model directory for PBench SFT (symlinks)
  local src="${SFT_CHECKPOINT_DIR}/${GR1_TRANSFORMER_SUBDIR}"
  [[ -d "${src}" ]] || { echo "missing SFT transformer directory: ${src}" >&2; exit 1; }
  [[ -f "${src}/config.json" ]] || { echo "missing transformer config: ${src}/config.json" >&2; exit 1; }
  [[ -d "${PRETRAIN_DIR}/text_encoder" ]] || { echo "missing text_encoder: ${PRETRAIN_DIR}/text_encoder" >&2; exit 1; }
  [[ -d "${PRETRAIN_DIR}/vae" ]] || { echo "missing vae: ${PRETRAIN_DIR}/vae" >&2; exit 1; }
  mkdir -p "${PBENCH_SFT_MODEL_DIR}"
  ln -sfn "${src}" "${PBENCH_SFT_MODEL_DIR}/transformer"
  ln -sfn "${PRETRAIN_DIR}/text_encoder" "${PBENCH_SFT_MODEL_DIR}/text_encoder"
  ln -sfn "${PRETRAIN_DIR}/vae" "${PBENCH_SFT_MODEL_DIR}/vae"
}

run_pbench_one() {  # $1=run_name  $2=save_dir  $3=frames  $4=model_dir(empty=pretrain)
  local run_name="$1" save_dir="$2" frames="$3" model_dir="${4:-}"
  echo "------------------------------------------------------------"
  echo "PBench generation: ${run_name}  (NUM_FRAMES=${frames}, $(python3 -c "print(round(${frames}/${FPS},4))")s)"
  if generation_done "${save_dir}" "${PBENCH_EXPECTED}"; then
    echo "  already complete, skipping."; return
  fi
  if [[ -d "${save_dir}" && ( -f "${save_dir}/run.log" || -f "${save_dir}/submit.log" ) ]]; then
    echo "  found a task in progress; waiting instead of resubmitting."
    wait_for_generation "${run_name}" "${save_dir}" "${PBENCH_EXPECTED}"; return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [DRY_RUN] would submit: MODEL_DIR='${model_dir}' NUM_FRAMES=${frames} SAVE_DIR=${save_dir}"; return
  fi
  MODE="full" \
  RUN_NAME="${run_name}" \
  OUTPUT_ROOT="$(dirname "${save_dir}")" \
  SAVE_DIR="${save_dir}" \
  MODEL_DIR="${model_dir}" \
  DATA_PATH="${PBENCH_DATA_PATH}" \
  DATA_LIMIT="0" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  GPU_IDS="${GPU_IDS}" \
  ALLOW_MULTI_GPU_SMOKE="${ALLOW_MULTI_GPU_SMOKE}" \
  NUM_FRAMES="${frames}" \
  FPS="${FPS}" \
  HEIGHT="${PBENCH_HEIGHT}" \
  WIDTH="${PBENCH_WIDTH}" \
  SEED="${SEED}" \
  GPU_MEMORY_SAMPLES="${save_dir}/gpu_memory_samples.csv" \
  GPU_MEMORY_PEAK="${save_dir}/gpu_memory_peak.json" \
  "${EVEWORLD_ROOT}/benchmarks/pbench/launch_pbench_robot_kjob.sh"
  wait_for_generation "${run_name}" "${save_dir}" "${PBENCH_EXPECTED}"
}

run_dreamgen_one() {  # $1=model_key(sft|pretrain)  $2=checkpoint_dir  $3=use_ema  $4=label  $5=frames
  local model_key="$1" ckpt="$2" use_ema="$3" label="$4" frames="$5"
  local run_name="${model_key}_dreamgen_8gpu_${label}_full_${DREAMGEN_SWEEP_ID}"
  local save_dir="${DREAMGEN_OUTPUT_ROOT}/${run_name}"
  echo "------------------------------------------------------------"
  echo "DreamGen generation: ${run_name}  (NUM_FRAMES=${frames}, $(python3 -c "print(round(${frames}/${FPS},4))")s)"
  if generation_done "${save_dir}" "${DREAMGEN_EXPECTED}"; then
    echo "  already complete, skipping."; return
  fi
  if [[ -d "${save_dir}" && ( -f "${save_dir}/run.log" || -f "${save_dir}/submit.log" ) ]]; then
    echo "  found a task in progress; waiting instead of resubmitting."
    wait_for_generation "${run_name}" "${save_dir}" "${DREAMGEN_EXPECTED}"; return
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  [DRY_RUN] would submit: CHECKPOINT_DIR=${ckpt} USE_EMA=${use_ema} NUM_FRAMES=${frames} SAVE_DIR=${save_dir}"; return
  fi
  CHECKPOINT_DIR="${ckpt}" \
  PRETRAIN_DIR="${PRETRAIN_DIR}" \
  USE_EMA="${use_ema}" \
  EVAL_ROOT="${DREAMGEN_EVAL_ROOT}" \
  OUTPUT_ROOT="${DREAMGEN_OUTPUT_ROOT}" \
  RUN_NAME="${run_name}" \
  SAVE_DIR="${save_dir}" \
  SUMMARY_PATH="${save_dir}/generation_summary.json" \
  GPU_IDS="${GPU_IDS}" \
  NUM_FRAMES="${frames}" \
  FPS="${FPS}" \
  HEIGHT="${DREAMGEN_HEIGHT}" \
  WIDTH="${DREAMGEN_WIDTH}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  SEED="${SEED}" \
  "${EVEWORLD_ROOT}/benchmarks/dreamgenbench/launch_gr1_dreamgen_generation_kjob.sh"
  wait_for_generation "${run_name}" "${save_dir}" "${DREAMGEN_EXPECTED}"
}

cd "${REPO_DIR}"

echo "============================================================"
echo "Short-duration inference-only fill-in (no judge)"
echo "TASKS:            ${TASKS}"
echo "DreamGen slots:   ${LENGTHS}"
echo "PBench slots:     ${PBENCH_LENGTHS}"
echo "GPU_IDS:          ${GPU_IDS}"
echo "Steps/FPS/Seed:   ${NUM_INFERENCE_STEPS} / ${FPS} / ${SEED}"
echo "DRY_RUN:          ${DRY_RUN}"
echo "============================================================"

for task in ${TASKS}; do
  case "${task}" in
    pbench_pretrain)
      echo; echo "### task: PBench Pretrain"
      for item in ${PBENCH_LENGTHS}; do
        label="${item%%:*}"; frames="${item##*:}"
        run_name="pbench_robot_${label}_full_${PBENCH_PRETRAIN_SWEEP_ID}"
        run_pbench_one "${run_name}" "${PBENCH_PRETRAIN_GEN_ROOT}/${run_name}" "${frames}" ""
      done
      ;;
    pbench_sft)
      echo; echo "### task: PBench GR1/SFT"
      prepare_sft_model_dir
      for item in ${PBENCH_LENGTHS}; do
        label="${item%%:*}"; frames="${item##*:}"
        run_name="gr1_pbench_robot_${label}_full_${PBENCH_SFT_SWEEP_ID}"
        run_pbench_one "${run_name}" "${PBENCH_SFT_GEN_ROOT}/${run_name}" "${frames}" "${PBENCH_SFT_MODEL_DIR}"
      done
      ;;
    dreamgen_pretrain)
      echo; echo "### task: DreamGen Pretrain"
      for item in ${LENGTHS}; do
        label="${item%%:*}"; frames="${item##*:}"
        run_dreamgen_one "pretrain" "${PRETRAIN_DIR}" "0" "${label}" "${frames}"
      done
      ;;
    dreamgen_sft)
      echo; echo "### task: DreamGen GR1/SFT"
      for item in ${LENGTHS}; do
        label="${item%%:*}"; frames="${item##*:}"
        run_dreamgen_one "sft" "${SFT_CHECKPOINT_DIR}" "1" "${label}" "${frames}"
      done
      ;;
    *)
      echo "Unknown TASK: ${task} (choose: pbench_pretrain pbench_sft dreamgen_pretrain dreamgen_sft)" >&2
      exit 1
      ;;
  esac
done

echo
echo "============================================================"
echo "All requested tasks finished. Generation outputs:"
echo "  PBench Pretrain: ${PBENCH_PRETRAIN_GEN_ROOT}/pbench_robot_<duration>_full_${PBENCH_PRETRAIN_SWEEP_ID}"
echo "  PBench SFT:      ${PBENCH_SFT_GEN_ROOT}/gr1_pbench_robot_<duration>_full_${PBENCH_SFT_SWEEP_ID}"
echo "  DreamGen:        ${DREAMGEN_OUTPUT_ROOT}/{pretrain,sft}_dreamgen_8gpu_<duration>_full_${DREAMGEN_SWEEP_ID}"
echo "============================================================"
