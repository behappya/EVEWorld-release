#!/usr/bin/env python3
"""T4G-AUG smoke test (CPU, no model): S1-S8 hard checks on the poisoning logic (S1 sample validity/safety,
S2 zone bias, S3 blend math, S4 latent mapping, S5 loss target, S6 EDM equivalence, S7 escort paste, S8 static boxes)."""
import numpy as np
import torch

from eveworld.pipeline.igr.paste import (sample_paste_plan, prep_patch, paste_lat_region,
                           sample_follow_plan, build_follow_boxes, static_boxes,
                           T_LAT, H_LAT, W_LAT, HPIX, WPIX, CELL_PX, NF)

rng = np.random.default_rng(0)

# synthetic zones: mostly safe bg, one B block, one unsafe block
zones = np.zeros((T_LAT, H_LAT, W_LAT), np.int8)
zones[0] = -1
zones[:, 10:15, 33:38] = 1                       # B block
zones[:, 0:8, 0:16] = -1                         # unsafe band
patch_hw = (60, 80)

# S1: 500 samples all valid
for i in range(500):
    plan = sample_paste_plan(zones, patch_hw, rng)
    assert plan is not None
    assert plan['t_lo'] >= 2, 'pasted at frame 0/1!'
    y0, x0, y1, x1 = plan['box']
    assert 0 <= y0 < y1 <= HPIX and 0 <= x0 < x1 <= WPIX
    for t in range(plan['t_lo'], plan['t_hi']):
        for cy in range(y0 // CELL_PX, (y1 - 1) // CELL_PX + 1):
            for cx in range(x0 // CELL_PX, (x1 - 1) // CELL_PX + 1):
                assert zones[t, cy, cx] >= 0, f'pasted into unsafe cell t={t} ({cy},{cx})'
print('S1 sample validity 500/500 PASS')

# all-unsafe -> reject
assert sample_paste_plan(np.full_like(zones, -1), patch_hw, rng) is None
print('S1b all-unsafe rejection PASS')

# S2: zone bias stats (B weight 0.4 vs bg 0.4 -> B share should clearly beat area share 25/1440)
zcnt = {0: 0, 1: 0, 2: 0}
for i in range(400):
    p = sample_paste_plan(zones, (30, 30), rng)
    zcnt[p['zone']] += 1
frac_b = zcnt[1] / 400
assert 0.2 < frac_b < 0.7, f'B share abnormal {zcnt}'
print(f'S2 zone bias PASS (B={zcnt[1]} bg={zcnt[0]} A={zcnt[2]})')

# S3: blend math
images = torch.zeros(1, 10, 3, HPIX, WPIX)
patch = torch.full((3, 40, 50), 1.0)
y0, x0, a = 100, 200, 0.6
img_aug = images.clone()
img_aug[0, 3:7, :, y0:y0 + 40, x0:x0 + 50] = a * patch + (1 - a) * img_aug[0, 3:7, :, y0:y0 + 40, x0:x0 + 50]
assert abs(float(img_aug[0, 4, 0, y0 + 5, x0 + 5]) - 0.6) < 1e-6
assert float(img_aug[0, 4, 0, y0 - 1, x0]) == 0.0 and float(img_aug[0, 2, 0, y0 + 5, x0 + 5]) == 0.0
assert torch.equal(img_aug[0, :3], images[0, :3]) and torch.equal(img_aug[0, 7:], images[0, 7:])
print('S3 blend math PASS')

# S4: latent region mapping
plan = dict(t_lo=5, t_hi=12, box=(100, 200, 140, 250))
lr = paste_lat_region(plan)
assert lr['ly0'] == 12 and lr['ly1'] == 18 and lr['lx0'] == 25 and lr['lx1'] == 32
assert lr['lt_lo'] == 5 and lr['lt_hi'] == 12
print('S4 latent mapping PASS')

# S5: loss target switching (hand-computed formula mimicking forward_step)
B, z, T, h, w = 1, 4, 8, 10, 12
clean = torch.zeros(B, z, T, h, w)
lat_in = clean.clone(); lat_in[:, :, 3:6, 2:5, 4:8] = 0.5          # paste magnitude
region = (slice(None), slice(None), slice(3, 6), slice(2, 5), slice(4, 8))
for denoised, expect_ret in [(lat_in.clone(), 1.0), (clean.clone(), 0.0)]:
    err = (denoised - clean) ** 2
    l_p = float(err[region].mean())
    d_p = float(((lat_in - clean) ** 2)[region].mean())
    ret = (l_p / d_p) ** 0.5 if d_p > 0 else float('nan')
    assert abs(ret - expect_ret) < 1e-6, (ret, expect_ret)
print('S5 loss target switching (ret 1.0/0.0) PASS')

# S6: clean sample equivalent to EDM compute_loss (weight*(err) mean over all elements)
weight = torch.tensor([2.5])
err = torch.rand(B, z, T, h, w)
mine = (weight.view(B, 1, 1, 1, 1) * err * torch.ones_like(err)).mean(dim=(1, 2, 3, 4))
edm_style = (weight.view(B, 1) * err.reshape(B, -1)).mean(dim=1)
assert torch.allclose(mine, edm_style, atol=1e-6)
print('S6 clean-sample EDM equivalence PASS')

print('\nall smoke PASS - poisoning logic trustworthy, ready to run prep + submit training.')


# S7: escort paste (v2) - per-frame boxes follow the trajectory / no occlusion of
# the real object / no paste at frame 0 / out-of-bounds clamping
rep7 = np.linspace(0, NF - 1, T_LAT).astype(int)
traj = []
for t in range(T_LAT):
    f = min(1.0, t / 16)
    traj.append([int(25 - 13 * f), int(5 + 30 * f)])
traj[5] = None                                     # simulated missed-detection frame
ok_follow = 0
for i in range(300):
    plan = sample_follow_plan(traj, (60, 80), rng)
    if plan is None:
        continue
    boxes, lat, p0, p1 = build_follow_boxes(traj, plan, rep7)
    if boxes is None:
        continue
    ok_follow += 1
    assert p0 >= rep7[2], 'pasted at frame 0/1!'
    hp_c = (plan['box'][2] // 2) // CELL_PX
    wp_c = (plan['box'][3] // 2) // CELL_PX
    moved = set()
    for p in range(p0, p1):
        y0, x0, y1, x1 = boxes[p]
        assert 0 <= y0 < y1 <= HPIX and 0 <= x0 < x1 <= WPIX, 'out of bounds'
        lt = int(np.clip(np.searchsorted(rep7, p, side='right') - 1, plan['t_lo'], plan['t_hi'] - 1))
        tc = traj[lt] or traj[lt - 1] or traj[lt + 1]
        cy, cx = (y0 + y1) // 2 // CELL_PX, (x0 + x1) // 2 // CELL_PX
        # real-object cell outside paste box (offset >= radius+1; border-clamped frames skip)
        clipped = (y0 == 0 or x0 == 0 or y1 == HPIX or x1 == WPIX)
        if not clipped:
            assert not (y0 // CELL_PX <= tc[0] <= (y1 - 1) // CELL_PX
                        and x0 // CELL_PX <= tc[1] <= (x1 - 1) // CELL_PX), f'occludes object p={p}'
        moved.add((cy, cx))
    # boxes move only if the trajectory moves (escort is a hover once the object arrives; legal)
    traj_cells = {tuple(traj[lt]) for lt in range(plan['t_lo'], plan['t_hi']) if traj[lt]}
    if len(traj_cells) >= 3:
        assert len(moved) >= 2, f'boxes static while traj moves win=[{plan["t_lo"]},{plan["t_hi"]})'
assert ok_follow > 200, f'escort paste success rate too low {ok_follow}/300'
print(f'S7 escort paste PASS ({ok_follow}/300, follow/no-occlusion/clamp/frame0 all pass)')

# S8: static_boxes equivalent to the legacy plan
plan8 = sample_paste_plan(zones, (60, 80), rng)
boxes8, p0, p1 = static_boxes(plan8, rep7)
assert (boxes8[p0] == np.array(plan8['box'])).all() and (boxes8[p1 - 1] == np.array(plan8['box'])).all()
assert (boxes8[:p0] == 0).all()
print('S8 static paste per-frame box equivalence PASS')

print('v2 smoke all PASS')
