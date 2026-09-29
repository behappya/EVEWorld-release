#!/usr/bin/env python3
"""Copy-paste augmentation assets for the trainset_v1 clips -> wmb_adapt/aug_assets_v1/<name>.npz.

The probe module's grid/path constants are patched to 640x480 / W_LAT 40 before the prep module
is imported, so its from-import picks up the patched values.
"""
import glob
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

GAGI = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))
TRAIN = f"{GAGI}/wmb_adapt/trainset_v1"
ANNO = f"{GAGI}/wmb_adapt/t4g_anno"
OUT = f"{GAGI}/wmb_adapt/aug_assets_v1"

os.environ.setdefault("T4G_W_LAT", "40")
os.environ.setdefault("T4G_WPIX", "640")

from eveworld.pipeline.probe import ghost_probe as G  # noqa: E402

G.W_LAT = 40
G.WPIX = 640
G.VIDEO_ROOT = TRAIN
G.ANNO_DIR = ANNO

from eveworld.pipeline.igr import prep as AP  # noqa: E402  (from-import reads the patched values here)


def main():
    names = sorted(os.path.splitext(os.path.basename(p))[0]
                   for p in glob.glob(f"{TRAIN}/*.mp4"))
    names = [n for n in names if not os.path.exists(f"{OUT}/{n}.npz")]
    if not names:
        print("[w6] all assets already exist, nothing to fill in")
        return
    argv = ["w6_aug_prep",
            "--out-dir", OUT,
            "--anno-dir", ANNO,
            "--video-root", TRAIN,
            "--vids", *names]
    for k in ("--shard-index", "--num-shards", "--device"):
        if k in sys.argv:
            i = sys.argv.index(k)
            argv += [k, sys.argv[i + 1]]
    sys.argv = argv
    AP.main()


if __name__ == "__main__":
    main()
