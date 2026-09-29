#!/usr/bin/env python3
"""T4G-EMPTY-MAP: safety probe for the L_ghost "should-be-empty mask" (note 44, CPU).

punishable(c,t; theta,Delta) = GT motion (grayscale diff of adjacent rep frames) over
[t-Delta, t+Delta] all < theta; GDINO cross-checks slow-motion misses only.
"""
import argparse
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np

from eveworld.pipeline.probe import probe as P
from eveworld.pipeline.probe.ghost_probe import cell_motion, T_LAT, H_LAT, W_LAT, HPIX, WPIX, NF, CELL_PX, ANNO_DIR, VIDEO_ROOT

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

THETAS = [1.5, 3.0, 5.0, 8.0]
DELTAS = [0, 1, 2, 3]


def window_static(motion, theta, delta):
    """punishable[t,c] = motion over window [t-delta, t+delta] all < theta. (T,30,48) bool."""
    hit = motion >= theta
    out = np.zeros_like(hit)
    for t in range(T_LAT):
        lo, hi = max(0, t - delta), min(T_LAT, t + delta + 1)
        out[t] = ~hit[lo:hi].any(axis=0)
    return out


def analyze_one(vid):
    try:
        anno = json.load(open(os.path.join(ANNO_DIR, f'{vid}.json')))
        vp = os.path.join(VIDEO_ROOT, f'{vid}.mp4')
        if not os.path.exists(vp):
            return None
        frames = P.sample_frames_like_training(vp, NF, HPIX, WPIX)
        rep = np.linspace(0, NF - 1, T_LAT).astype(int)
        motion = cell_motion(frames, rep)
        per = anno['per_lat_frame']
        t_arr = anno['t_arrival']
        visited = [(t, tuple(fr['target_cell'])) for t, fr in enumerate(per) if fr['target_cell']]

        rec = {'vid': vid, 't_arrival': t_arr,
               'motion_q': [float(np.percentile(motion[1:], q)) for q in (50, 75, 90, 99)]}
        for theta in THETAS:
            for delta in DELTAS:
                pm = window_static(motion, theta, delta)
                area = float(pm[1:].mean())
                # B_pre retention
                bfrac = float('nan')
                if t_arr and t_arr >= 4:
                    tot = kept = 0
                    for t in range(1, max(2, int(t_arr * 0.75))):
                        for c in per[t]['b_cells']:
                            tot += 1
                            kept += bool(pm[t, c[0], c[1]])
                    bfrac = kept / tot if tot else float('nan')
                # slow-motion miss: visited (c, t±delta+1) yet punishable
                leak = tot_v = 0
                for tv, c in visited:
                    for t in range(max(1, tv - delta - 1), min(T_LAT, tv + delta + 2)):
                        tot_v += 1
                        leak += bool(pm[t, c[0], c[1]])
                rec[f'th{theta}_d{delta}'] = dict(
                    area=area, b_pre_keep=bfrac,
                    visited_leak=leak / tot_v if tot_v else float('nan'))
        return rec
    except Exception as e:
        return {'vid': vid, 'error': str(e)[:200]}


def overlay(vid, out_dir, theta=3.0, delta=2):
    anno = json.load(open(os.path.join(ANNO_DIR, f'{vid}.json')))
    frames = P.sample_frames_like_training(os.path.join(VIDEO_ROOT, f'{vid}.mp4'), NF, HPIX, WPIX)
    rep = np.linspace(0, NF - 1, T_LAT).astype(int)
    motion = cell_motion(frames, rep)
    pm = window_static(motion, theta, delta)
    per = anno['per_lat_frame']
    rows = []
    for t in (3, 8, 14, 20):
        im = np.ascontiguousarray(frames[rep[t]].copy())
        red = im.copy()
        for gy in range(H_LAT):
            for gx in range(W_LAT):
                if pm[t, gy, gx]:
                    red[gy * CELL_PX:(gy + 1) * CELL_PX, gx * CELL_PX:(gx + 1) * CELL_PX, 0] = 255
        im = (0.65 * im + 0.35 * red).astype(np.uint8)
        bc = per[t]['b_cells']
        if bc:
            ys = [c[0] for c in bc]; xs = [c[1] for c in bc]
            cv2.rectangle(im, (min(xs) * CELL_PX, min(ys) * CELL_PX),
                          ((max(xs) + 1) * CELL_PX, (max(ys) + 1) * CELL_PX), (40, 40, 255), 2)
        tc = per[t]['target_cell']
        if tc:
            cv2.circle(im, (tc[1] * CELL_PX + 8, tc[0] * CELL_PX + 8), 10, (40, 255, 40), 2)
        cv2.putText(im, f't={t} t*={anno["t_arrival"]} red=punishable', (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        rows.append(im)
    cv2.imwrite(os.path.join(out_dir, f'overlay_{vid}_th{theta}_d{delta}.png'),
                cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=f'{GAGI}/eve_v2_outputs/track4gen_probe/empty_map')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--n-overlay', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    vids = sorted([f[:-5] for f in os.listdir(ANNO_DIR) if f[0].isdigit()], key=int)
    with Pool(a.workers) as pool:
        recs = [r for r in pool.map(analyze_one, vids) if r]
    errs = [r for r in recs if 'error' in r]
    recs = [r for r in recs if 'error' not in r]

    summary = {'n': len(recs), 'errors': errs, 'table': {}}
    print(f'\n===== T4G-EMPTY-MAP (n={len(recs)}, err={len(errs)}) =====')
    print('motion percentiles p50/p75/p90/p99 (grayscale): ',
          [round(float(np.median([r['motion_q'][i] for r in recs])), 2) for i in range(4)])
    print(f'{"theta":>6} {"Delta":>5} | {"area":>8} | {"B_pre":>9} | {"leak":>8}')
    for theta in THETAS:
        for delta in DELTAS:
            k = f'th{theta}_d{delta}'
            area = float(np.median([r[k]['area'] for r in recs]))
            bk = float(np.nanmedian([r[k]['b_pre_keep'] for r in recs]))
            lk = float(np.nanmedian([r[k]['visited_leak'] for r in recs]))
            lk_max = float(np.nanmax([r[k]['visited_leak'] for r in recs]))
            summary['table'][k] = dict(area_med=area, b_pre_keep_med=bk, leak_med=lk, leak_max=lk_max)
            print(f'{theta:>6} {delta:>5} | {area:>8.3f} | {bk:>9.3f} | {lk:>5.3f}/max{lk_max:.3f}')
    print('\nReading: area>0.3 and B_pre>0.8 and leak~0 -> usable (theta,Delta) mask tiers;')
    print('         leak>0.05 -> layer a GDINO corridor exemption.')
    print('         Eyeball the overlay PNG before writing conclusions into docs.')
    summary['per_video'] = recs
    json.dump(summary, open(os.path.join(a.out_dir, 'empty_map_summary.json'), 'w'), indent=1)
    # human-eye overlay: evenly sample n videos with t_arrival
    cand = [r['vid'] for r in recs if r['t_arrival'] is not None]
    pick = cand[::max(1, len(cand) // a.n_overlay)][:a.n_overlay]
    for v in pick:
        overlay(v, a.out_dir)
    print(f'overlay PNG x{len(pick)} + summary -> {a.out_dir}')


if __name__ == '__main__':
    main()
