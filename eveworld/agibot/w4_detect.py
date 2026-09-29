#!/usr/bin/env python3
"""GDINO detection over the trainset_v1 clips -> wmb_adapt/t4g_anno/<name>.json (GR1 schema).

The prompt is the native instruction from <name>.txt.orig (simple nouns help parse_objects); the
monkey-patch moves the GR1 constants to 640x480 / W_LAT 40.
"""
import argparse
import glob
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

GAGI = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))
TRAIN = f"{GAGI}/wmb_adapt/trainset_v1"
OUT_DEFAULT = f"{GAGI}/wmb_adapt/t4g_anno"

from eveworld.pipeline.annotate import detect as D  # noqa: E402

# WMB convention patch: 640x480, latent 30x40 (VAE 8x + patch 16 -> 640/16=40, 480/16=30)
D.VIDEO_ROOT = TRAIN
D.WIMG = 640
D.HIMG = 480
D.W_LAT = 40
D.H_LAT = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=OUT_DEFAULT)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from eveworld.pipeline.annotate.gdino import GDinoLocator

    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{TRAIN}/*.mp4"))
    names = names[a.shard_index::a.num_shards]
    loc = GDinoLocator(device=a.device)
    print(f"[wmb-detect] shard {a.shard_index}/{a.num_shards}: {len(names)} videos", flush=True)
    for name in names:
        out = f"{a.out_dir}/{name}.json"
        if os.path.exists(out):
            continue
        orig = f"{TRAIN}/{name}.txt.orig"
        prompt = open(orig).read().strip() if os.path.exists(orig) else open(f"{TRAIN}/{name}.txt").read().strip()
        try:
            anno = D.process(loc, name, prompt)
            json.dump(anno, open(out, "w"), ensure_ascii=False)
            print(f"  {name}: gate={anno['gate_enabled']}({anno.get('gate_reason')}) "
                  f"det={anno.get('n_detected_frames')}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {name} FAIL: {str(e)[:120]}", flush=True)
    print("[wmb-detect] shard done", flush=True)


if __name__ == "__main__":
    main()
