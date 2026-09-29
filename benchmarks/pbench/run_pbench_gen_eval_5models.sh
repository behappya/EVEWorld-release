#!/usr/bin/env bash
# PBench Robot Domain (VQA) evaluation: judge 5 representative models under pbench_gen in parallel.
# Judge: Qwen/Qwen3.6-35B-A3B @ 127.0.0.1:8000, 80 concurrent requests per model (400 total).
set -uo pipefail

cd third_party/giga-world-0

GEN="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/pbench_gen"
OUT_ROOT="${GAGI_ROOT:-$HOME/gagi}/eve_v2_outputs/pbench_gen_eval/qwen_vqa"
MODELS="pretrain round0 t4g_wmapA_pre_seed42_s250 t4g_wmapA_pre_noaug_s50 t4g_wmaponly_s150"

mkdir -p "${OUT_ROOT}"

pids=()
for m in ${MODELS}; do
  VIDEO_DIR="${GEN}/pbench_robot_3p8s_${m}" \
  EVAL_DIR="${OUT_ROOT}/${m}_qwen36_12" \
  RUN_LOG="${OUT_ROOT}/${m}_qwen36_12/eval.log" \
  QWEN_BASE=http://127.0.0.1:8000/v1 \
  QWEN_MODEL=Qwen/Qwen3.6-35B-A3B \
  CONCURRENCY=80 MAX_INFLIGHT=80 \
  bash benchmarks/pbench/eval_pbench_robot_qwen_vqa.sh > "${OUT_ROOT}/${m}_launcher.log" 2>&1 &
  pids+=($!)
  echo "started ${m} pid=$!"
done

fail=0
for p in "${pids[@]}"; do
  wait "${p}" || fail=$((fail+1))
done

echo "===== ALL EVAL DONE (failed_procs=${fail}) ====="
for m in ${MODELS}; do
  s="${OUT_ROOT}/${m}_qwen36_12/qwen_vqa_summary.json"
  if [ -f "${s}" ]; then
    echo "--- ${m} ---"
    python3 -c "import json;d=json.load(open('${s}'));print('  submitted=%s completed=%s errors=%s micro=%s macro/Domain=%s'%(d.get('submitted'),d.get('completed'),d.get('build_error_count'),round(d.get('question_micro_accuracy',d.get('micro_accuracy',0)),4),round(d.get('sample_macro_accuracy',d.get('macro_accuracy',0)),4)))" 2>/dev/null || echo "  (summary parse failed, see ${s})"
  else
    echo "--- ${m}: no summary ---"
  fi
done
