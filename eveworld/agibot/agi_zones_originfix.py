#!/usr/bin/env python3
"""Raise safe (zone 0) cells within radius 2 of the target's initial cell to zone 2, once the
target has left it (>3 cells for 2 consecutive frames). TRANSFER clips only, 0->2 only; the npz
is rewritten in place keeping patch/box.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

import agi_aug_prep as APP  # noqa: E402  (performs all the monkey-patching + agi_build_zones)
from eveworld.pipeline.probe import probe as P  # noqa: E402
from eveworld.pipeline.probe.ghost_probe import cell_motion  # noqa: E402

CLEAN = APP.CLEAN
ANNO = APP.ANNO
OUT = APP.OUT
T_LAT, H_LAT, W_LAT = 24, 30, 40
NF, HPIX, WPIX = 93, 480, 640
LEAVE_DIST, ORIGIN_R, SETTLE = 3, 2, 2


def origin_patch(zones, anno):
    c0 = anno.get('target_cell_0')
    if anno.get('category') != 'TRANSFER' or not c0:
        return zones, 0
    per = anno['per_lat_frame']
    t_leave = None
    for t in range(1, T_LAT - 1):
        c, c2 = per[t].get('target_cell'), per[t + 1].get('target_cell')
        if c and c2 and max(abs(c[0] - c0[0]), abs(c[1] - c0[1])) > LEAVE_DIST \
                and max(abs(c2[0] - c0[0]), abs(c2[1] - c0[1])) > LEAVE_DIST:
            t_leave = t
            break
    if t_leave is None:
        return zones, 0
    n = 0
    gy0, gx0 = int(c0[0]), int(c0[1])
    for t in range(min(t_leave + SETTLE, T_LAT), T_LAT):
        for gy in range(max(0, gy0 - ORIGIN_R), min(H_LAT, gy0 + ORIGIN_R + 1)):
            for gx in range(max(0, gx0 - ORIGIN_R), min(W_LAT, gx0 + ORIGIN_R + 1)):
                if zones[t, gy, gx] == 0:
                    zones[t, gy, gx] = 2
                    n += 1
    return zones, n


def one(name):
    try:
        anno = json.load(open(f'{ANNO}/{name}.json'))
        fp = f'{OUT}/{name}.npz'
        # materialize the old arrays before overwriting the same file (np.load is lazy)
        with np.load(fp) as d:
            keep = {k: d[k].copy() for k in ('patch', 'box') if k in d}
        rep = np.linspace(0, NF - 1, T_LAT).astype(int)
        frames = P.sample_frames_like_training(f'{CLEAN}/{name}.mp4', NF, HPIX, WPIX)
        zones = APP.agi_build_zones(anno, cell_motion(frames, rep))
        zones, n = origin_patch(zones, anno)
        np.savez_compressed(fp, zones=zones, **keep)
        return name, anno.get('skill', '?'), int((zones == 2).any()), n, 'ok'
    except Exception as e:  # noqa: BLE001
        return name, '?', 0, 0, f'FAIL {type(e).__name__}: {str(e)[:80]}'


def main():
    from collections import Counter
    from multiprocessing import Pool
    names = sorted(f[:-5] for f in os.listdir(ANNO)
                   if f.endswith('.json') and not f.startswith('_'))
    fixed, fails = 0, 0
    has_a = Counter()
    with Pool(16) as pool:
        for i, (name, sk, ha, n, st) in enumerate(pool.imap_unordered(one, names)):
            if st != 'ok':
                fails += 1
                print(f'  {name}: {st}', flush=True)
                continue
            has_a[sk] += ha
            fixed += int(n > 0)
            if (i + 1) % 150 == 0:
                print(f'  {i + 1}/{len(names)} origin_fixed={fixed}', flush=True)
    print(f'[originfix] done: {len(names)} clips, origin-A added in {fixed}, fail={fails}')
    print('hasA by skill:', dict(has_a))


if __name__ == '__main__':
    main()
