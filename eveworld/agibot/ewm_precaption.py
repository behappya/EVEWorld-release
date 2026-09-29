#!/usr/bin/env python3
"""EWMBench caption pre-run (zero GPU): query the QWEN_BASE endpoint for models whose layout
conversion is done and cache <model>_caption_responses.json, so the GPU eval chain can skip
semantics.
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

GAGIBENCH = os.environ.get("GAGIBENCH_ROOT", os.path.expanduser("~/gagibench"))
sys.path.insert(0, f"{GAGIBENCH}/EWMBench")
from EWMBench.caption import inference_api, prepare_prompt  # noqa: E402
import json_repair  # noqa: E402

GAGI = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))
LAYOUT = f"{GAGI}/eve_v2_outputs/ewmbench_gen/eval_layout"
SAVE = f"{GAGI}/eve_v2_outputs/ewmbench_eval"
EP = os.environ.get("QWEN_BASE", "http://127.0.0.1:8000/v1")


def one(args):
    key, vdir = args
    try:
        return key, json_repair.loads(inference_api(EP, vdir, prepare_prompt(vdir)))
    except Exception as e:  # noqa: BLE001
        return key, "Error: " + str(e)


def main(models):
    for m in models:
        out_dir = f"{SAVE}/{m}"
        os.makedirs(out_dir, exist_ok=True)
        out = f"{out_dir}/{m}_caption_responses.json"
        if os.path.exists(out):
            print(f"[skip] {m}")
            continue
        jobs = []
        for vdir in sorted(glob.glob(f"{LAYOUT}/{m}_dataset/*/*/*/video")):
            parts = vdir.split("/")
            task, ep, trial = parts[-4], parts[-3], parts[-2]
            jobs.append((f"{m}_dataset_{task}_{ep}_{trial}", vdir))
        if len(jobs) != 63:
            print(f"[warn] {m}: {len(jobs)} dirs (63 expected)")
        res = {}
        with ThreadPoolExecutor(64) as ex:
            for k, v in ex.map(one, jobs):
                res[k] = v
        bad = sum(1 for v in res.values() if not isinstance(v, dict))
        json.dump(res, open(out, "w"), indent=4)
        print(f"[ok] {m}: {len(res)} entries, non-dict {bad}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
