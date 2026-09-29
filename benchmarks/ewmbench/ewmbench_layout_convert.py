#!/usr/bin/env python3
"""Convert GW-0 EWMBench side-by-side mp4s into the official EWMBench evaluation layout.

Crops the right half and writes frame_%05d.jpg under <gen_root>/eval_layout/<model>_dataset/
(seed42->1, seed43->2, seed44->3, matching the official generated_samples tree).
"""
import argparse
import glob
import os

import cv2

GEN_ROOT = f"{os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))}/eve_v2_outputs/ewmbench_gen"
SEED_TO_SAMPLE = {"seed42": "1", "seed43": "2", "seed44": "3"}
GEN_W = 640  # generated view width (right half)


def convert_one(mp4_path: str, out_dir: str) -> int:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {mp4_path}")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gen = frame[:, -GEN_W:]
        cv2.imwrite(os.path.join(out_dir, f"frame_{n:05d}.jpg"),
                    gen, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        n += 1
    cap.release()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", default=GEN_ROOT)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--expect-frames", type=int, default=93)
    args = ap.parse_args()

    problems = []
    for model in args.models:
        for seed_dir, sample_id in SEED_TO_SAMPLE.items():
            mp4s = sorted(glob.glob(f"{args.gen_root}/{model}/{seed_dir}/*.mp4"))
            if not mp4s:
                problems.append(f"{model}/{seed_dir}: no mp4")
                continue
            for mp4 in mp4s:
                rid = os.path.splitext(os.path.basename(mp4))[0]  # <task>_<episode>
                task, episode = rid.split("_", 1)
                out_dir = (f"{args.gen_root}/eval_layout/{model}_dataset/"
                           f"{task}/{episode}/{sample_id}/video")
                if len(glob.glob(f"{out_dir}/frame_*.jpg")) == args.expect_frames:
                    continue  # already converted
                n = convert_one(mp4, out_dir)
                if n != args.expect_frames:
                    problems.append(
                        f"{rid} {seed_dir}: {n} frames (expected {args.expect_frames})")
                print(f"[ok] {model}/{seed_dir}/{rid} -> {n} frames", flush=True)

    if problems:
        print("\n[WARN] bad entries:")
        for p in problems:
            print("  ", p)
        return 1
    print("\nall conversions done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
