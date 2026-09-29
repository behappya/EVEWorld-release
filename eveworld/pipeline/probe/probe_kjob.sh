#!/usr/bin/env bash
#SBATCH --job-name=t4g_probe
#SBATCH --gpus-per-task=nvidia.com/gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# EVE · Track4Gen feature-traceability probe kjob payload (single node, video sharding
# across 8 GPUs by default).
# Pure probe: export 28 feature layers + compute metrics (motion-group traceability /
# static-group stability), no training.
# Multi-GPU orchestration lives in Python (eveworld/pipeline/probe/probe.py --dispatch-gpus N); kjobctl's
# slurm interpreter does not support bash background-job syntax, hence no & / wait.
# NGPU=1 degrades to a single GPU over all videos.
set -uo pipefail

# refuse accidental runs outside a submitted job (discipline: no local GPU runs).
if [[ -z "${SLURM_JOB_ID:-}" && "${ALLOW_LOCAL_RUN:-0}" != "1" ]]; then
  echo "Refusing to run outside a submitted job. Submit via eveworld/pipeline/probe/launch_probe_kjob.sh" >&2
  exit 2
fi

# passthrough: export each trailing KEY=VALUE (same mechanism as kjob_eve_eag_generate.sh).
for arg in "$@"; do
  [[ "${arg}" == *=* ]] && export "${arg}" || { echo "bad arg: ${arg}" >&2; exit 1; }
done

REPO_DIR="${REPO_DIR:-${EVEWORLD_ROOT}/third_party/giga-world-0}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-EVEWorld}"
TRAIN_VENV="${TRAIN_VENV:-${GAGI_ROOT:-$HOME/gagi}/envs/giga_world_train_venv}"
TRAIN_PYTHON="${TRAIN_PYTHON:-${TRAIN_VENV}/bin/python}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
# probe default base = anmix_s200 (current best, candidate loss-training base).
MODEL_DIR="${MODEL_DIR:-${GAGI}/eve_v2_outputs/anchor_models/probe_anmix_s200}"
TRANSFORMER="${TRANSFORMER:-${MODEL_DIR}/transformer}"
VAE="${VAE:-${MODEL_DIR}/vae}"
TEXT_ENCODER="${TEXT_ENCODER:-${MODEL_DIR}/text_encoder}"

VIDEO_ROOT="${VIDEO_ROOT:-${GAGI}/gr1_finetune_data/raw_data}"
# 12-20 real GT clips, including shelf vertical-transport tasks (13/32/76).
VIDEO_IDS="${VIDEO_IDS:-13,32,76,14,15,16,17,18,19,20,21,23,24,25,26,27}"
SIGMAS="${SIGMAS:-0.25,0.7,2.0,5.0}"      # low/mid/high noise levels (Track4Gen)
NUM_FRAMES="${NUM_FRAMES:-93}"
HEIGHT="${HEIGHT:-480}"; WIDTH="${WIDTH:-768}"; FPS="${FPS:-16}"
DTYPE="${DTYPE:-bf16}"
NGPU="${NGPU:-8}"                          # GPUs per node (1-8), video sharding
EPE_GO="${EPE_GO:-2.0}"                    # criterion: endpoint error < 2 cells
STAB_GO="${STAB_GO:-0.90}"                 # criterion: static-group self-similarity > 0.9
P_HI="${P_HI:-85}"; P_LO="${P_LO:-40}"
OUT_DIR="${OUT_DIR:-${GAGI}/eve_v2_outputs/track4gen_probe/anmix_s200}"

# shellcheck disable=SC1090
source "${CONDA_SH}"; conda activate "${CONDA_ENV}"
export PYTHONPATH="${EVEWORLD_ROOT}:${REPO_DIR}:${EVEWORLD_ROOT}/third_party/giga-models:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "${EVEWORLD_ROOT}/eveworld/pipeline"

echo "=========================================="
echo " Track4Gen probe  host=$(hostname)"
echo "  model    = ${TRANSFORMER}"
echo "  videos   = ${VIDEO_IDS}"
echo "  sigmas   = ${SIGMAS}   frames=${NUM_FRAMES} ${HEIGHT}x${WIDTH}"
echo "  ngpu     = ${NGPU}   dtype=${DTYPE}"
echo "  judge    = EPE<${EPE_GO} cell & STAB>${STAB_GO}"
echo "  out      = ${OUT_DIR}"
echo "=========================================="
nvidia-smi || true

"${TRAIN_PYTHON}" probe/probe.py \
  --transformer "${TRANSFORMER}" --vae "${VAE}" --text-encoder "${TEXT_ENCODER}" \
  --video-root "${VIDEO_ROOT}" --video-ids "${VIDEO_IDS}" \
  --sigmas "${SIGMAS}" \
  --num-frames "${NUM_FRAMES}" --height "${HEIGHT}" --width "${WIDTH}" --fps "${FPS}" \
  --dtype "${DTYPE}" --device cuda \
  --p-hi "${P_HI}" --p-lo "${P_LO}" --epe-go "${EPE_GO}" --stab-go "${STAB_GO}" \
  --out-dir "${OUT_DIR}" \
  --dispatch-gpus "${NGPU}"
RC=$?

echo "[t4g] payload done rc=${RC} -> ${OUT_DIR}/t4g_probe_summary.json"
exit "${RC}"
