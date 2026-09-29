#!/usr/bin/env python3
"""T4G-EXAM v2: improved GDINO persistence exam -- fixes robot-arm gripper false detections and
artifact noise: box_thr 0.35, gripper-overlap drop (>35%), dedup 60px (was 40px in v1).

Criteria: inv0 = frame-0 count (0 -> dropped); DUP = >=2 consecutive frames with valid_count >
inv0; VANISH = >=3 consecutive frames with count == 0 in [2,22).
"""
import argparse
import json
import os
import numpy as np

from eveworld.pipeline.probe import probe as P
from eveworld.pipeline.annotate.gdino import GDinoLocator, parse_objects
from eveworld.pipeline.annotate.detect import detect_all

NF, HPIX, WPIX, T_LAT = 93, 480, 768, 24


def compute_overlap(box_obj, box_grip):
    """Overlap ratio between object box and gripper box (intersection / object area)."""
    x0 = max(box_obj[0], box_grip[0])
    y0 = max(box_obj[1], box_grip[1])
    x1 = min(box_obj[2], box_grip[2])
    y1 = min(box_obj[3], box_grip[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_obj = max(1.0, (box_obj[2] - box_obj[0]) * (box_obj[3] - box_obj[1]))
    return inter / area_obj


def count_valid_instances(obj_dets, gripper_dets, min_dist=60, grip_overlap_thr=0.35):
    """Drop gripper FPs and near artifacts, then count physically independent instances."""
    valid_centers = []
    grip_boxes = [g[2] for g in gripper_dets]

    for (cx, cy, box, sc) in obj_dets:
        # rule 1: gripper-overlap filter (if the gripper covers >35% of the box, it is a
        # gripper false positive or a contact artifact)
        if any(compute_overlap(box, gbox) > grip_overlap_thr for gbox in grip_boxes):
            continue

        # rule 2: spatial-independence dedup (center distance > 60px)
        if all(abs(cx - a) + abs(cy - b) > min_dist for a, b in valid_centers):
            valid_centers.append((cx, cy))

    return len(valid_centers)


def exam_video_v2(loc, path, prompt):
    frames = P.sample_frames_like_training(path, NF, HPIX, WPIX)
    name = parse_objects(prompt)['mover']
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)

    counts = []
    for t in rep:
        frame_rgb = frames[int(t)]
        # 1. raise the object detection gate to 0.35
        obj_dets = detect_all(loc, frame_rgb, name, topk=6, box_thr=0.35)
        # 2. detect the robot-arm gripper
        gripper_dets = detect_all(loc, frame_rgb, "robot gripper", topk=3, box_thr=0.15)

        valid_c = count_valid_instances(obj_dets, gripper_dets)
        counts.append(valid_c)

    inv0 = counts[0]
    if inv0 == 0:
        return dict(undetectable=True, counts=counts, inv0=0, dup=False, vanish=False,
                    dup_frames=0, zero_frames=0, max_count=max(counts))

    # DUP criterion: >=2 consecutive frames with valid_count > inv0
    dup = False
    run = 0
    dup_frames = 0
    for c in counts:
        if c > inv0:
            run += 1
            dup_frames += 1
            if run >= 2:
                dup = True
        else:
            run = 0

    # VANISH criterion: >=3 consecutive zeros in frames [2,22)
    vanish = False
    run = 0
    zero_frames = 0
    for t in range(2, min(22, T_LAT)):
        if counts[t] == 0:
            run += 1
            zero_frames += 1
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
    print(f'[exam_v2] shard {a.shard_index}/{a.num_shards}: {len(items)} videos', flush=True)

    recs = []
    for i, (arm, path, prompt) in enumerate(items):
        try:
            r = exam_video_v2(loc, path, prompt)
            r.update(arm=arm, video=os.path.basename(path))
            recs.append(r)
            if (i + 1) % 20 == 0:
                print(f'  {i+1}/{len(items)}', flush=True)
        except Exception as e:
            print(f'  FAIL {path}: {str(e)[:100]}', flush=True)

    json.dump(recs, open(a.out, 'w'))
    print(f'[exam_v2] shard {a.shard_index} -> {a.out} ({len(recs)} recs)', flush=True)


if __name__ == '__main__':
    main()
