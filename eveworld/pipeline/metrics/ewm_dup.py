#!/usr/bin/env python3
"""EWMBench DUP detection: manual mover table (EWMBench phrasing breaks parse_objects) + right-half
crop (side-by-side videos) + inv0 from the GT condition image; DUP/vanish criteria match
eveworld.pipeline.metrics.exam_v2.
Run: EWM_MODELS="pretrain round0 ..." python ewm_dup.py <shard> <nshard> <out.json>
"""
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

from eveworld.pipeline.probe import probe as P
from eveworld.pipeline.annotate.gdino import GDinoLocator
from eveworld.pipeline.annotate.detect import detect_all
from eveworld.pipeline.metrics.exam_v2 import count_valid_instances, NF, HPIX, WPIX, T_LAT

GAGI = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))
GEN = f"{GAGI}/eve_v2_outputs/ewmbench_gen"
GT = f"{GAGI}/ewmbench_data/gt_dataset"

# hand-picked mover per episode (the object to detect); episodes are grouped by task above
MOVER = {
    "367_649524": "toast", "367_649559": "toast", "367_650191": "toast",
    "392_651464": "cup brush", "392_664600": "cup brush", "392_681186": "cup brush",
    "497_766602": "freezer door", "497_773025": "freezer door", "497_773496": "freezer door",
    "511_743247": "showerhead", "511_743964": "showerhead", "511_744776": "showerhead",
    "543_798615": "detergent bottle", "543_798749": "detergent bottle", "543_807480": "detergent bottle",
    "558_787136": "golden kettle", "558_789120": "transparent teapot", "558_791059": "transparent teapot",
    "574_808158": "golden kettle", "574_824748": "blue hat kettle", "574_834014": "blue hat kettle",
}
# DUP applicability (door / multi-bottle episodes are False; lets aggregation group them)
APPLICABLE = {k: True for k in MOVER}
for k in MOVER:
    if MOVER[k] in ("freezer door", "detergent bottle"):
        APPLICABLE[k] = False


def right_half_frames(mp4):
    cap = cv2.VideoCapture(mp4)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        w = f.shape[1]
        frames.append(cv2.cvtColor(f[:, w // 2:], cv2.COLOR_BGR2RGB))
    cap.release()
    # T_LAT frames evenly spaced in time
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, T_LAT).astype(int)
    return [frames[i] for i in idx]


def inv0_cond(loc, task, ep, mover):
    p = f"{GT}/{task}/{ep}/prompt/init_frame.png"
    if not os.path.exists(p):
        return 0
    img = np.array(Image.open(p).convert("RGB"))
    obj = detect_all(loc, img, mover, topk=6, box_thr=0.35)
    grip = detect_all(loc, img, "robot gripper", topk=3, box_thr=0.15)
    return count_valid_instances(obj, grip)


def exam(loc, mp4, task, ep, mover):
    frames = right_half_frames(mp4)
    if not frames:
        return dict(undetectable=True, err="no_frames")
    counts = []
    for fr in frames:
        obj = detect_all(loc, fr, mover, topk=6, box_thr=0.35)
        grip = detect_all(loc, fr, "robot gripper", topk=3, box_thr=0.15)
        counts.append(count_valid_instances(obj, grip))
    inv0 = inv0_cond(loc, task, ep, mover) or (max(counts[:3]) if counts else 0)
    if inv0 == 0:
        return dict(undetectable=True, counts=counts, inv0=0, dup=False, vanish=False, max_count=max(counts))
    run = mx = 0
    for c in counts:
        run = run + 1 if c > inv0 else 0
        mx = max(mx, run)
    dup = mx >= 2
    zrun = zmx = 0
    for t in range(2, min(22, len(counts))):
        zrun = zrun + 1 if counts[t] == 0 else 0
        zmx = max(zmx, zrun)
    return dict(undetectable=False, counts=counts, inv0=inv0, dup=dup, vanish=zmx >= 3, max_count=max(counts))


def main():
    shard, nshard, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    models = os.environ["EWM_MODELS"].split()
    jobs = []
    for m in models:
        for seed in ("seed42", "seed43", "seed44"):
            for name, mover in MOVER.items():
                task, ep = name.split("_")
                mp4 = f"{GEN}/{m}/{seed}/{name}.mp4"
                if os.path.exists(mp4):
                    jobs.append((m, seed, name, task, ep, mover, mp4))
    jobs = jobs[shard::nshard]
    loc = GDinoLocator(device="cuda")
    res = []
    for i, (m, seed, name, task, ep, mover, mp4) in enumerate(jobs):
        r = exam(loc, mp4, task, ep, mover)
        r.update(model=m, seed=seed, episode=name, mover=mover, applicable=APPLICABLE[name])
        res.append(r)
        if i % 20 == 0:
            print(f"[shard{shard}] {i}/{len(jobs)}", flush=True)
    json.dump(res, open(out, "w"))


if __name__ == "__main__":
    main()
