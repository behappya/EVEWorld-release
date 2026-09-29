#!/usr/bin/env python3
"""T4G-AUG pure poisoning logic (no training-stack dependency; smoke-tested directly;
reused by trainer/transform)."""
import os

import cv2
import numpy as np

# WMB adaptation (issue 56): grid width / pixel width overridable via env vars
# (GR1 default 768->48; WMB 640->40)
T_LAT, H_LAT = 24, 30
W_LAT = int(os.environ.get('T4G_W_LAT', '48'))
NF, HPIX, CELL_PX = 93, 480, 16
WPIX = int(os.environ.get('T4G_WPIX', '768'))
VAE_SP = 8                                  # pixel -> VAE latent space ratio


def sample_paste_plan(zones, patch_hw, rng, zone_bias=(0.4, 0.2, 0.4), k_retry=60,
                      min_dur=3):
    """zones (24,30,48) int8 (-1 unsafe/0 bg/1 B/2 A); patch_hw=(hp,wp).
    Returns plan { t_lo, t_hi, box (pixel), alpha, scale, blur, zone } or None; constraints: t_lo>=2, all patch-covered cells zones>=0 over the window, zone by bias."""
    hp0, wp0 = patch_hw
    for _ in range(k_retry):
        t_lo = int(rng.integers(2, T_LAT - min_dur + 1))
        t_hi = T_LAT if rng.random() < 0.5 else int(rng.integers(t_lo + min_dur, T_LAT + 1))
        scale = float(rng.uniform(0.8, 1.2))
        hp, wp = max(12, int(hp0 * scale)), max(12, int(wp0 * scale))
        if hp > HPIX // 2 or wp > WPIX // 2:
            continue
        win = zones[t_lo:t_hi]
        ok_all = (win >= 0).all(axis=0)
        if not ok_all.any():
            continue
        tag0 = zones[t_lo]
        cand = {z: np.argwhere(ok_all & (tag0 == z)) for z in (1, 2, 0)}
        zs, ps = [], []
        for z, p in zip((1, 2, 0), (zone_bias[0], zone_bias[1], zone_bias[2])):
            if len(cand[z]):
                zs.append(z); ps.append(p)
        if not zs:
            continue
        ps = np.array(ps) / sum(ps)
        z = zs[int(rng.choice(len(zs), p=ps))]
        gy, gx = cand[z][int(rng.integers(len(cand[z])))]
        cy, cx = gy * CELL_PX + CELL_PX // 2, gx * CELL_PX + CELL_PX // 2
        y0 = int(np.clip(cy - hp // 2, 0, HPIX - hp)); x0 = int(np.clip(cx - wp // 2, 0, WPIX - wp))
        y1, x1 = y0 + hp, x0 + wp
        cys = range(y0 // CELL_PX, (y1 - 1) // CELL_PX + 1)
        cxs = range(x0 // CELL_PX, (x1 - 1) // CELL_PX + 1)
        if not all(ok_all[yy, xx] for yy in cys for xx in cxs):
            continue
        return dict(t_lo=t_lo, t_hi=t_hi, box=(y0, x0, y1, x1),
                    alpha=float(rng.uniform(0.4, 1.0)), scale=scale,
                    blur=bool(rng.random() < 0.3), zone=int(z))
    return None


def prep_patch(patch_u8, plan):
    """Resize/blur the patch per plan -> uint8 (hp,wp,3)."""
    y0, x0, y1, x1 = plan['box']
    p = cv2.resize(patch_u8, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    if plan['blur']:
        p = cv2.GaussianBlur(p, (0, 0), 1.5)
    return p


def paste_lat_region(plan):
    """Latent indices of the paste window (frame domain & VAE 8x space)."""
    y0, x0, y1, x1 = plan['box']
    return dict(lt_lo=plan['t_lo'], lt_hi=plan['t_hi'],
                ly0=y0 // VAE_SP, ly1=min(HPIX // VAE_SP, (y1 + VAE_SP - 1) // VAE_SP),
                lx0=x0 // VAE_SP, lx1=min(WPIX // VAE_SP, (x1 + VAE_SP - 1) // VAE_SP))


# v2: escort paste (closes the "gripper birth channel" loophole) - the fake patch follows
# the object/gripper track at a fixed offset, frame by frame.
def static_boxes(plan, rep):
    """Per-frame boxes of the static paste (93,4): same box in the window, 0 outside."""
    boxes = np.zeros((NF, 4), np.int64)
    p0 = int(rep[plan['t_lo']])
    p1 = NF if plan['t_hi'] >= T_LAT else int(rep[plan['t_hi']])
    boxes[p0:p1] = plan['box']
    return boxes, p0, p1


def sample_follow_plan(target_cells, patch_hw, rng, min_dur=3):
    """Escort-paste plan: the offset is patch radius + 1..3 cells, keeping the real
    object uncovered. target_cells: len24 list."""
    n_known = sum(1 for c in target_cells if c)
    if n_known < 8:
        return None
    hp0, wp0 = patch_hw
    scale = float(rng.uniform(0.8, 1.2))
    hp, wp = max(12, int(hp0 * scale)), max(12, int(wp0 * scale))
    if hp > HPIX // 2 or wp > WPIX // 2:
        return None
    # safe offset = ceil(patch radius in cells) + 1..3 cells (flooring can hit a
    # real-object cell on bad boundary alignment)
    ry = -(-(hp // 2) // CELL_PX) + 1 + int(rng.integers(0, 3))
    rx = -(-(wp // 2) // CELL_PX) + 1 + int(rng.integers(0, 3))
    off = (ry * (1 if rng.random() < 0.5 else -1), rx * (1 if rng.random() < 0.5 else -1))
    t_lo = int(rng.integers(2, T_LAT - min_dur + 1))
    t_hi = T_LAT if rng.random() < 0.5 else int(rng.integers(t_lo + min_dur, T_LAT + 1))
    return dict(t_lo=t_lo, t_hi=t_hi, box=(0, 0, hp, wp), alpha=float(rng.uniform(0.4, 1.0)),
                scale=scale, blur=bool(rng.random() < 0.3), zone=3, offset=off, mode='follow')


def build_follow_boxes(target_cells, plan, rep):
    """Escort-paste per-frame boxes (93,4) + loose latent region. Missing-track frames
    fall back to the nearest earlier/later value."""
    t_lo, t_hi = plan['t_lo'], plan['t_hi']
    hp, wp = plan['box'][2], plan['box'][3]
    centers, obj_pos, last = {}, {}, None
    for lt in range(t_lo, t_hi):
        c = target_cells[lt] or last
        if c is None:
            for lt2 in range(lt + 1, t_hi):
                if target_cells[lt2]:
                    c = target_cells[lt2]
                    break
        if c is None:
            return None, None, None, None
        last = c
        obj_pos[lt] = c                                          # object pos estimate (fill-in)
        centers[lt] = (c[0] + plan['offset'][0], c[1] + plan['offset'][1])
    boxes = np.zeros((NF, 4), np.int64)
    p0 = int(rep[t_lo])
    p1 = NF if t_hi >= T_LAT else int(rep[t_hi])
    lys, lxs = [], []
    for p in range(p0, p1):
        lt = int(np.clip(np.searchsorted(rep, p, side='right') - 1, t_lo, t_hi - 1))
        cy, cx = centers[lt]
        py, px = cy * CELL_PX + CELL_PX // 2, cx * CELL_PX + CELL_PX // 2
        y0 = int(np.clip(py - hp // 2, 0, HPIX - hp))
        x0 = int(np.clip(px - wp // 2, 0, WPIX - wp))
        # hard check: clamped paste box never covers a real-object cell (filled estimate)
        # -> reject
        tc = obj_pos[lt]
        if (y0 // CELL_PX <= tc[0] <= (y0 + hp - 1) // CELL_PX
                and x0 // CELL_PX <= tc[1] <= (x0 + wp - 1) // CELL_PX):
            return None, None, None, None
        boxes[p] = (y0, x0, y0 + hp, x0 + wp)
        lys += [y0, y0 + hp]; lxs += [x0, x0 + wp]
    lat = dict(lt_lo=t_lo, lt_hi=t_hi,
               ly0=min(lys) // VAE_SP, ly1=min(HPIX // VAE_SP, (max(lys) + VAE_SP - 1) // VAE_SP),
               lx0=min(lxs) // VAE_SP, lx1=min(WPIX // VAE_SP, (max(lxs) + VAE_SP - 1) // VAE_SP))
    return boxes, lat, p0, p1
