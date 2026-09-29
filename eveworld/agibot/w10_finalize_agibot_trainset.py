#!/usr/bin/env python3
"""Task-balanced downsampling of the EWMBench-adapted trainset -> agibot_ewm_train_final/.

Per-task cap --cap (default 150), overflow downsampled with a fixed seed; writes mp4+txt
hardlinks plus the finalize manifest.
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

GAGI = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))
SRC = f"{GAGI}/agibot_ewm_train"
DST = f"{GAGI}/agibot_ewm_train_final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    os.makedirs(DST, exist_ok=True)

    by_task = defaultdict(lambda: defaultdict(list))  # task -> ep -> [name]
    for p in sorted(glob.glob(f"{SRC}/*.mp4")):
        name = os.path.splitext(os.path.basename(p))[0]
        task, ep, seg = name.split("_")
        by_task[task][ep].append(name)

    manifest = {}
    for task, eps in sorted(by_task.items()):
        # round-robin sampling: segment 1 of every episode first, then segment 2, ... up to CAP
        # (episode diversity first)
        ep_list = list(eps.keys())
        random.shuffle(ep_list)
        picked = []
        rnd = 0
        while len(picked) < a.cap:
            added = False
            for ep in ep_list:
                segs = sorted(eps[ep])
                if rnd < len(segs):
                    picked.append(segs[rnd])
                    added = True
                    if len(picked) >= a.cap:
                        break
            if not added:
                break
            rnd += 1
        manifest[task] = sorted(picked)
        for name in picked:
            for ext in (".mp4", ".txt"):
                d = f"{DST}/{name}{ext}"
                if not os.path.exists(d):
                    os.link(f"{SRC}/{name}{ext}", d)

    total = sum(len(v) for v in manifest.values())
    json.dump({"cap": a.cap, "seed": a.seed, "total": total,
               "per_task": {k: len(v) for k, v in manifest.items()},
               "files": manifest},
              open(f"{DST}/_finalize_manifest.json", "w"), indent=1)
    print("finalized distribution:", {k: len(v) for k, v in manifest.items()}, "| total:", total)


if __name__ == "__main__":
    main()
