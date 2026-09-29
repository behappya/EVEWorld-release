#!/usr/bin/env python3
"""Offline CPU scoring of the three EWMBench semantics metrics from the cached caption responses
(calls evaluate_runs_configs directly; the official compute_semantics tail raises NameError).
"""
import json
import os
import sys

GAGIBENCH = os.environ.get("GAGIBENCH_ROOT", os.path.expanduser("~/gagibench"))
sys.path.insert(0, f"{GAGIBENCH}/EWMBench")
from EWMBench.semantics import evaluate_runs_configs  # noqa: E402

GAGI = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))
SAVE = f"{GAGI}/eve_v2_outputs/ewmbench_eval"
CLIP = f"{GAGI}/ewmbench_ckpt/openai_clip-vit-base-patch16"
GT = f"{SAVE}/pretrain/gt_caption_responses.json"  # single global GT caption

CONFIGS = [
    {"metric_type": "BLEUScore", "key": "General", "bleu_n_gram": 4},
    {"metric_type": "CLIPScore", "key": "General"},
]


def logic_of(path):
    d = json.load(open(path))
    vals = []
    for v in d.values():
        if isinstance(v, dict) and "Overall_Constraints" in v:
            vals.append(1.0 if v["Overall_Constraints"] in (True, "true", "True") else 0.0)
    return sum(vals) / len(vals) if vals else None


def main(models):
    print(f"{'model':42s} {'BLEU':>7} {'CLIP':>7} {'logic':>6}")
    for m in models:
        cj = f"{SAVE}/{m}/{m}_caption_responses.json"
        try:
            r = evaluate_runs_configs(CONFIGS, cj, GT, CLIP)
            bleu = list(r["BLEUScore_ngram4"].values())[0]
            clip = list(r["CLIPScore"].values())[0]
            lg = logic_of(cj)
            print(f"{m:42s} {bleu:7.4f} {clip:7.3f} {lg:6.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{m:42s} FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
