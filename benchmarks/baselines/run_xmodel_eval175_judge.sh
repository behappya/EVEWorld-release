#!/usr/bin/env bash
# One-shot judge for xmodel eval175 (126 items): prepare audit + qwen_if/pa_i scoring.
# Needs all three xmodel generations complete (wan22_ti2v_5b / cogvideox15_5b_i2v / wan22_i2v_a14b, 126 each);
# Qwen judge at QWEN_BASE, same scoring rubric as the in-domain eval175 runs.
set -uo pipefail

source "${CONDA_SH:-$HOME/miniconda/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-gigaworld}"

GAGI="${GAGI_ROOT:-$HOME/gagi}"
REPO="${EVEWORLD_ROOT:-$(pwd)}"
GEN_ROOT=${GAGI}/gr1_dreamgen_eval/xmodel_eval175
OUT_ROOT=${GAGI}/eve_v2_outputs/xmodel_eval175_eval
EVAL_OUT=${GAGI}/gr1_dreamgen_eval/eval_outputs
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:8000/v1}"
CONCURRENCY="${CONCURRENCY:-80}"

cd ${REPO}/eveworld/evaluation

echo "===== 1. prepare audit: Wan, both durations (93f 768x480) ====="
python eval175_prepare.py \
  --generation-root "${GEN_ROOT}" \
  --output-root "${OUT_ROOT}" \
  --models wan22_ti2v_5b,wan22_i2v_a14b \
  --expected-frames 93 --expected-width 768 --expected-height 480 \
  --expected-fps 16 --expected-seed 42 --no-stage
rc1=$?

echo "===== 2. prepare audit: cogvideox (96f 1360x768, native output) ====="
python eval175_prepare.py \
  --generation-root "${GEN_ROOT}" \
  --output-root "${OUT_ROOT}_cogx" \
  --models cogvideox15_5b_i2v \
  --expected-frames 96 --expected-width 1360 --expected-height 768 \
  --expected-fps 16 --expected-seed 42 --no-stage
rc2=$?
echo "prepare rc: wan=${rc1} cogx=${rc2} (reports in each output_root/prepare_report.json; if the seed audit warns about summary layout, verify by hand and continue)"

echo "===== 3. parallel scoring for the three models (${CONCURRENCY} concurrent each) ====="
cd ${REPO}
pids=()
judge_one() {
  local model=$1 manifest=$2
  python benchmarks/dreamgenbench/eval_dreamgenbench_qwen_api.py \
    --manifest "${manifest}" \
    --output-root "${EVAL_OUT}" \
    --run-name "eval175_${model}_qwen36_12" \
    --qwen-base "${QWEN_BASE}" \
    --qwen-model Qwen/Qwen3.6-35B-A3B \
    --metrics qwen_if,pa_i \
    --concurrency "${CONCURRENCY}" --max-inflight "${CONCURRENCY}" \
    > "${EVAL_OUT}/eval175_${model}_qwen36_12_judge.log" 2>&1
}
judge_one wan22_ti2v_5b     "${OUT_ROOT}/manifests/wan22_ti2v_5b.jsonl" &
pids+=($!)
judge_one wan22_i2v_a14b    "${OUT_ROOT}/manifests/wan22_i2v_a14b.jsonl" &
pids+=($!)
judge_one cogvideox15_5b_i2v "${OUT_ROOT}_cogx/manifests/cogvideox15_5b_i2v.jsonl" &
pids+=($!)
fail=0
for p in "${pids[@]}"; do wait "${p}" || fail=$((fail+1)); done

echo "===== 4. summary (failed=${fail}) ====="
for m in wan22_ti2v_5b wan22_i2v_a14b cogvideox15_5b_i2v; do
  s="${EVAL_OUT}/eval175_${m}_qwen36_12_summary.json"
  if [ -f "$s" ]; then
    python3 -c "
import json; d=json.load(open('$s'))
for met,v in d.items():
    print(f'  ${m} {met}: {v.get(\"positive\")}/{v.get(\"count\")} = {v.get(\"score\"):.3f} errors={v.get(\"errors\")}')" 2>/dev/null || echo "  ${m}: failed to parse summary ($s)"
  else
    echo "  ${m}: no summary"
  fi
done
