#!/usr/bin/env python3
"""AgiBot dual-arm gripper detection over the 777 clips -> agibot_t4g_probe/gripper_anno/.

Side is the gripper center cell (gx < W_LAT/2 -> left), which the weightmap corridors use.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
CLEAN = f'{GAGI}/agibot_ewm_clean'
OUT_DEFAULT = f'{GAGI}/eve_v2_outputs/agibot_t4g_probe/gripper_anno'

from eveworld.pipeline.annotate import detect as D  # noqa: E402
from eveworld.pipeline.annotate import gripper_detect as G  # noqa: E402
import numpy as np  # noqa: E402

for mod in (D, G):
    mod.WIMG = 640
    mod.HIMG = 480
D.W_LAT = 40
D.H_LAT = 30
G.VIDEO_ROOT = CLEAN
W_LAT = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from eveworld.pipeline.annotate.gdino import GDinoLocator

    names = sorted(f[:-4] for f in os.listdir(CLEAN)
                   if f.endswith('.mp4') and not f.endswith('_trans.mp4'))
    names = names[a.shard_index::a.num_shards]
    if a.limit:
        names = names[:a.limit]
    loc = GDinoLocator(device=a.device)
    print(f'[agi-gripper] shard {a.shard_index}/{a.num_shards}: {len(names)} clips', flush=True)
    for name in names:
        out = f'{a.out_dir}/{name}.json'
        if os.path.exists(out):
            continue
        try:
            anno = G.process(loc, name)
            for per in anno['per_lat_gripper']:
                for g in per:
                    g['side'] = 'left' if g['center'][1] < W_LAT // 2 else 'right'
            json.dump(anno, open(out, 'w'))
            avg = np.mean([len(p) for p in anno['per_lat_gripper']])
            print(f'  {name}: det={anno["n_det_frames"]}/24 avg={avg:.1f}', flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'  {name} FAIL: {str(e)[:120]}', flush=True)
    print('[agi-gripper] shard done', flush=True)


if __name__ == '__main__':
    main()
