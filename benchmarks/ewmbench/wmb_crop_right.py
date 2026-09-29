#!/usr/bin/env python3
"""WMB robotics postprocess: crop the right half of a side-by-side mp4 into a generated-only mp4.

Writes <gen_root>/eval_videos/<model>/<name>.mp4 (640x480, 16fps); output names must match the
first_frame names that evaluation.py looks up.
"""
import argparse
import glob
import os

import cv2

GEN_ROOT = f"{os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))}/eve_v2_outputs/wmb_robotics_gen"
GEN_W = 640


def crop_one(src: str, dst: str, fps: int = 16) -> int:
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    writer = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gen = frame[:, -GEN_W:]
        if writer is None:
            h, w = gen.shape[:2]
            writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        writer.write(gen)
        n += 1
    cap.release()
    if writer is not None:
        writer.release()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default=GEN_ROOT)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--expect-frames", type=int, default=93)
    args = ap.parse_args()

    problems = []
    for model in args.models:
        mp4s = sorted(glob.glob(f"{args.gen_root}/{model}/*.mp4"))
        if len(mp4s) != 50:
            problems.append(f"{model}: {len(mp4s)} mp4 (expected 50)")
        for src in mp4s:
            name = os.path.basename(src)
            dst = f"{args.gen_root}/eval_videos/{model}/{name}"
            if os.path.exists(dst):
                cap = cv2.VideoCapture(dst)
                nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                if nf == args.expect_frames:
                    continue
                os.remove(dst)
            n = crop_one(src, dst)
            if n != args.expect_frames:
                problems.append(f"{model}/{name}: {n} frames (expected {args.expect_frames})")
            print(f"[ok] {model}/{name} {n} frames", flush=True)

    if problems:
        print("\n[WARN] problems:")
        for p in problems:
            print("  ", p)
        return 1
    print("\nall crops done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
