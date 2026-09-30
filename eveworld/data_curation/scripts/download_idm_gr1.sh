#!/usr/bin/env bash
# Download the in-domain GR1 IDM weights (DreamGen/GR00T official, seonghyeonye/IDM_gr1): anchor layer of the TEA evaluation (IDM recovers actions from generated video -> jerk/OOD executability).
# Idempotent + resumable; DEST overrides the target dir and DRY_RUN=1 prints the plan only.
set -euo pipefail

REPO_ID="${REPO_ID:-seonghyeonye/IDM_gr1}"
DEST="${DEST:-${GAGI_ROOT:-$HOME/gagi}/idm_gr1_probe}"
# Uncomment the next line if your network needs a mirror (or export it before invoking):
# export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Prefer the hf CLI shipped with the gigaworld env (its Python has huggingface_hub);
# avoid ~/.local/bin/hf, whose shebang points at the system /usr/bin/python3 without huggingface_hub.
HF_BIN=""
for cand in \
  "${HF_BIN_OVERRIDE:-}" \
  "${HOME}/miniconda/envs/gigaworld/bin/hf" \
  "${HOME}/miniconda/envs/gigaworld/bin/huggingface-cli" \
  "hf" "huggingface-cli"; do
  [[ -z "$cand" ]] && continue
  if command -v "$cand" >/dev/null 2>&1; then HF_BIN="$cand"; break; fi
done
# Fallback: download via the gigaworld python -m huggingface_hub
GM_PY="${EVEWORLD_PYTHON:-$HOME/miniconda/envs/gigaworld/bin/python}"
if [[ -z "$HF_BIN" ]]; then
  if [[ -x "$GM_PY" ]] && "$GM_PY" -c "import huggingface_hub" 2>/dev/null; then
    HF_BIN="$GM_PY -m huggingface_hub.commands.huggingface_cli"
  else
    echo "[ERR] no usable hf CLI, and the gigaworld env has no huggingface_hub." >&2
    echo "      fix: ~/miniconda/envs/gigaworld/bin/pip install -U 'huggingface_hub[cli]'" >&2
    exit 1
  fi
fi

echo "=========================================="
echo " EVE - download IDM_gr1"
echo "  repo   : $REPO_ID"
echo "  dest   : $DEST"
echo "  hf bin : $HF_BIN"
echo "  mirror : ${HF_ENDPOINT:-<default hf.co>}"
echo "=========================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY_RUN] would run:"
  echo "  $HF_BIN download $REPO_ID --repo-type model --local-dir $DEST"
  exit 0
fi

mkdir -p "$DEST"

# hf download resumes by default and skips files that already exist with a matching hash, so it is safe to rerun.
# No --include: pull the whole repo (config/weights/experiment_cfg) and merge with the local files.
echo "[*] downloading (resumable, existing files are skipped)..."
"$HF_BIN" download "$REPO_ID" --repo-type model --local-dir "$DEST"

echo ""
echo "[*] self-check: are the weight files present"
# HF PreTrainedModel weights are *.safetensors or pytorch_model*.bin
shopt -s nullglob
weights=( "$DEST"/*.safetensors "$DEST"/pytorch_model*.bin "$DEST"/model*.safetensors )
if [[ ${#weights[@]} -eq 0 ]]; then
  echo "[ERR] no weight file found (*.safetensors / pytorch_model*.bin)." >&2
  echo "      current contents of $DEST:" >&2
  ls -la "$DEST" >&2
  echo "      likely causes: network/auth failure, or the repo keeps weights in a subdirectory." >&2
  exit 2
fi

echo "[OK] weight files:"
for w in "${weights[@]}"; do
  sz=$(du -h "$w" | cut -f1)
  echo "     $w  ($sz)"
done

echo ""
echo "[*] key files:"
for f in config.json experiment_cfg/conf.yaml experiment_cfg/metadata.json; do
  if [[ -e "$DEST/$f" ]]; then echo "     [ok] $f"; else echo "     [missing] $f  <- check"; fi
done

echo ""
echo "[DONE] IDM_gr1 is ready. Loading:"
echo "   from gr00t.model.idm import IDM"
echo "   idm = IDM.from_pretrained('$DEST')"
echo "   # or the official pipeline: python IDM_dump/dump_idm_actions.py --checkpoint '$DEST' ..."
