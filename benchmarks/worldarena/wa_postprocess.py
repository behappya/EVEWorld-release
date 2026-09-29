#!/usr/bin/env python3
"""WorldArena generation postprocess: side-by-side mp4 -> official evaluation layout.

Crops the generated half to 640x480, keeps the first 121 frames (official length; the
generator emits 125), and writes <out_root>/<model>_test/fixed_scene_task_episodeK.mp4 at 24fps.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import cv2

GEN = f"{os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))}/eve_v2_outputs/worldarena_gen"
OUT = f"{os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))}/eve_v2_outputs/worldarena_eval_videos"
GEN_W, FPS, KEEP = 640, 24, 121


def one(args):
    src, dst = args
    try:
        if os.path.exists(dst):
            cap = cv2.VideoCapture(dst)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n == KEEP:
                return "skip"
            os.remove(dst)
        cap = cv2.VideoCapture(src)
        vw = None
        n = 0
        while n < KEEP:
            ok, f = cap.read()
            if not ok:
                break
            g = f[:, -GEN_W:]
            if vw is None:
                vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                                     (g.shape[1], g.shape[0]))
            vw.write(g)
            n += 1
        cap.release()
        if vw:
            vw.release()
        return "ok" if n == KEEP else f"short:{n}"
    except Exception as e:  # noqa: BLE001
        return f"FAIL:{e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    jobs = []
    for m in a.models:
        od = f"{OUT}/{m}_test"
        os.makedirs(od, exist_ok=True)
        for src in sorted(glob.glob(f"{GEN}/{m}/*.mp4")):
            jobs.append((src, f"{od}/{os.path.basename(src)}"))
    print(f"jobs to process: {len(jobs)}")
    stats = {}
    with Pool(12) as p:
        for st in p.imap_unordered(one, jobs):
            stats[st.split(":")[0]] = stats.get(st.split(":")[0], 0) + 1
    print(stats)


if __name__ == "__main__":
    main()
