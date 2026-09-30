#!/usr/bin/env bash
set -euo pipefail

# Download Qwen/Qwen2.5-VL-7B-Instruct into the shared HF cache for the official
# DreamGenBench Qwen IF / PA-I eval (the official code looks the model up by name).

ROOT_DIR="${ROOT_DIR:-${GAGI_ROOT:-$HOME/gagi}}"
HF_HOME="${HF_HOME:-${ROOT_DIR}/.hf_home}"
HF_XET_CACHE="${HF_XET_CACHE:-${ROOT_DIR}/.hf_xet_cache}"
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-VL-7B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-${EVEWORLD_PYTHON:-$HOME/miniconda/envs/gigaworld/bin/python}}"

export HF_HOME
export HF_XET_CACHE

echo "DreamGenBench Qwen2.5-VL-7B downloader"
echo "Model repo:   ${MODEL_REPO}"
echo "HF_HOME:      ${HF_HOME}"
echo "HF_XET_CACHE: ${HF_XET_CACHE}"
echo
echo "If interrupted, run the same command again."

"${PYTHON_BIN}" -m huggingface_hub.cli.hf download "${MODEL_REPO}" --repo-type model

echo
echo "Download command finished."
