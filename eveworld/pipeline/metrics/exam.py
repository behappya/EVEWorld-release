#!/usr/bin/env python3
"""T4G-EXAM: GDINO persistence health check -- inv0 / DUP (>=2 consecutive frames with count >
inv0) / VANISH (>=3 consecutive frames with count == 0 in [2,22)); inv0 == 0 -> dropped.

Dedup: centers <=40px apart are one instance; box_thr=0.20 (same as eveworld.pipeline.annotate.detect).
Sharded via eveworld/pipeline/metrics/exam_dispatch.py; each --list-file line is "arm\tvideo_path\tprompt".
"""
import argparse
import json
import os

import numpy as np

from eveworld.pipeline.probe import probe as P
from eveworld.pipeline.annotate.gdino import GDinoLocator, parse_objects
from eveworld.pipeline.annotate.detect import detect_all

NF, HPIX, WPIX, T_LAT = 93, 480, 768, 24


def count_instances(dets, min_dist=40):
    centers = []
    for (cx, cy, box, sc) in dets:
        if all(abs(cx - a) + abs(cy - b) > min_dist for a, b in centers):
            centers.append((cx, cy))
    return len(centers)


def exam_video(loc, path, prompt):
    frames = P.sample_frames_like_training(path, NF, HPIX, WPIX)
    name = parse_objects(prompt)['mover']
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    counts = []
    for t in rep:
        dets = detect_all(loc, frames[int(t)], name, topk=6)
        counts.append(count_instances(dets))
    inv0 = counts[0]
    if inv0 == 0:
        return dict(undetectable=True, counts=counts, inv0=0, dup=False, vanish=False,
                    dup_frames=0, zero_frames=0, max_count=max(counts))
    # DUP: >=2 consecutive frames above inventory
    dup = False; run = 0; dup_frames = 0
    for c in counts:
        if c > inv0:
            run += 1; dup_frames += 1
            if run >= 2:
                dup = True
        else:
            run = 0
    # VANISH: >=3 consecutive zeros within [2,22)
    vanish = False; run = 0; zero_frames = 0
    for t in range(2, min(22, T_LAT)):
        if counts[t] == 0:
            run += 1; zero_frames += 1
            if run >= 3:
                vanish = True
        else:
            run = 0
    return dict(undetectable=False, counts=counts, inv0=inv0, dup=dup, vanish=vanish,
                dup_frames=dup_frames, zero_frames=zero_frames, max_count=max(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list-file', required=True)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    items = []
    for line in open(a.list_file):
        parts = line.rstrip('\n').split('\t')
        if len(parts) == 3:
            items.append(parts)
    items = items[a.shard_index::a.num_shards]
    loc = GDinoLocator(device=a.device)
    print(f'[exam] shard {a.shard_index}/{a.num_shards}: {len(items)} videos', flush=True)
    recs = []
    for i, (arm, path, prompt) in enumerate(items):
        try:
            r = exam_video(loc, path, prompt)
            r.update(arm=arm, video=os.path.basename(path))
            recs.append(r)
            if (i + 1) % 20 == 0:
                print(f'  {i+1}/{len(items)}', flush=True)
        except Exception as e:
            print(f'  FAIL {path}: {str(e)[:100]}', flush=True)
    json.dump(recs, open(a.out, 'w'))
    print(f'[exam] shard {a.shard_index} -> {a.out} ({len(recs)} recs)', flush=True)


if __name__ == '__main__':
    main()
