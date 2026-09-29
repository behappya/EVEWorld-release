#!/usr/bin/env python3
"""T4G-GHOST/EMPTY-MAP smoke (CPU, no model): synthetic anno+motion+dev to verify the probe logic.
"""
import numpy as np
import torch

from eveworld.pipeline.probe.ghost_probe import build_zones, zone_means, cell_dev, T_LAT, H_LAT, W_LAT
from eveworld.pipeline.probe.empty_map import window_static

# synthetic anno: object (25,5)->(12,35) straight line, t_arr=16
t_arr = 16
per = []
for t in range(T_LAT):
    f = min(1.0, t / t_arr)
    cy, cx = int(25 - 13 * f), int(5 + 30 * f)
    per.append({'target_cell': [cy, cx],
                'b_cells': [[y, x] for y in range(10, 15) for x in range(33, 38)],
                'detected': True})
anno = {'t_arrival': t_arr, 'per_lat_frame': per, 'distractor_cells': [[5, 5]],
        'a_cells': [[y, x] for y in range(23, 28) for x in range(3, 8)]}

# synthetic motion: object path cells high motion at pass-through time, static elsewhere
motion = np.ones((T_LAT, H_LAT, W_LAT), np.float32) * 0.5
for t in range(T_LAT):
    c = per[t]['target_cell']
    motion[t, max(0, c[0] - 1):c[0] + 2, max(0, c[1] - 1):c[1] + 2] = 20.0

zones = build_zones(anno, motion)

# S1: B_pre frame domain correct
assert zones['B_pre'], 'B_pre empty!'
assert all(t < 0.75 * t_arr + 1 for (t, y, x) in zones['B_pre']), 'B_pre has late frames'
assert all(t >= t_arr for (t, y, x) in zones['B_post']), 'B_post has early frames'
assert all((y, x) in {(yy, xx) for yy in range(10, 15) for xx in range(33, 38)}
           for (t, y, x) in zones['B_pre']), 'B_pre outside box'
# at arrival the target is inside B; its neighborhood must not appear in B_pre (cur_nb excluded)
for (t, y, x) in zones['B_pre']:
    c = per[t]['target_cell']
    assert max(abs(y - c[0]), abs(x - c[1])) > 2, f'B_pre contains target neighborhood t={t}'
print('S1 build_zones zoning correct PASS')

# S2: increasing field -> bins increasing
dev = np.zeros((T_LAT, H_LAT, W_LAT), np.float32)
for t in range(T_LAT):
    dev[t] = t / T_LAT
zm = zone_means(dev, zones)
b = [zm[f'Bbin_bin{i}'] for i in range(4) if zm[f'Bbin_bin{i}'] == zm[f'Bbin_bin{i}']]
assert len(b) >= 3 and all(b[i] < b[i + 1] for i in range(len(b) - 1)), f'bins not increasing: {b}'
print(f'S2 temporal bins increasing visible PASS {["%.3f" % x for x in b]}')

# S3: higher dev closer to visit time -> dt profile decreasing
dev2 = np.zeros_like(dev)
vis = {}
for t in range(T_LAT):
    c = per[t]['target_cell']
    vis.setdefault((c[0], c[1]), []).append(t)
for (cy, cx), ts in vis.items():
    for t in range(T_LAT):
        dt = min(abs(t - x) for x in ts)
        dev2[t, cy, cx] = max(0.0, 1.0 - 0.12 * dt)
zm2 = zone_means(dev2, zones)
p = [zm2[f'path_dt{d}'] for d in range(1, 9) if zm2[f'path_dt{d}'] == zm2[f'path_dt{d}']]
assert len(p) >= 4 and p[0] > p[-1], f'path profile not decreasing: {p}'
print(f'S3 path-smear profile measurable PASS dt1={p[0]:.3f} dt_last={p[-1]:.3f}')

# S4: cell_dev shape+values
x0 = torch.zeros(16, T_LAT, 60, 96)
xh = torch.full((16, T_LAT, 60, 96), 0.25)
d = cell_dev(xh, x0)
assert d.shape == (T_LAT, H_LAT, W_LAT) and abs(float(d.mean()) - 0.25) < 1e-6, d.shape
print('S4 cell_dev pooling correct PASS')

# S5: window_static exemption
mo = np.zeros((T_LAT, 4, 4), np.float32)
mo[10, 1, 1] = 99.0                       # t=10: this cell moves
for delta in (0, 2):
    from eveworld.pipeline.probe import empty_map as EM
    old = EM.T_LAT
    pm = window_static(mo, theta=3.0, delta=delta)
    banned = [t for t in range(T_LAT) if not pm[t, 1, 1]]
    assert banned == list(range(10 - delta, 10 + delta + 1)), (delta, banned)
assert window_static(mo, 3.0, 1)[:, 0, 0].all(), 'static cell wrongly exempted'
print('S5 window_static time-margin exemption correct PASS')

# S6: empty zone nan
zm3 = zone_means(dev, {'empty': []})
assert zm3['empty'] != zm3['empty'], 'nan handling failed'
print('S6 empty zone nan PASS')

print('\nall smoke tests PASS - probe logic trustworthy, safe to submit GPU job.')
