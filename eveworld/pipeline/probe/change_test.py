#!/usr/bin/env python3
"""T4G-CHANGE-TEST: validate idea 1 (change detection) -- the live-or-dead verdict.
Criterion: floor - self_at > 0 and self_pre - self_at > 0 -> duplication = detectable.
"""
import argparse
import csv
import json
import os
import re

import numpy as np

from eveworld.pipeline.probe import probe as P
from eveworld.pipeline.annotate.gdino import GDinoLocator
from eveworld.pipeline.annotate.detect import detect_all, px_to_cell

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
KP = os.environ.get('EVEWORLD_KAPPA_PACK', 'kappa_pack')  # human-annotation pack, not part of this release
T_LAT = 24


def parse_obj(instr):
    m = re.search(r'pick up (?:the )?(.+?)(?: from| to|\.|$)', instr, re.I)
    return m.group(1).strip() if m else instr


def self_sim(fnorm, t, cell):
    """cos(feat[t,cell], feat[t-1,cell]), unit-normalized."""
    return float((fnorm[t, cell[0], cell[1]] * fnorm[t - 1, cell[0], cell[1]]).sum())


def run(args):
    import torch
    from giga_models.nn import EDMLoss
    dtype, device = torch.bfloat16, 'cuda'
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)
    models = P.load_models(f'{args.model_dir}/transformer', f'{args.model_dir}/vae',
                           f'{args.model_dir}/text_encoder', device, dtype)
    loc = GDinoLocator(device=device)
    gold = list(csv.DictReader(open(f'{KP}/gold_labels_v1.csv')))
    key = {r['video_id']: r for r in csv.DictReader(open(f'{KP}/hidden_key_DO_NOT_SHOW_ANNOTATORS.csv'))}
    rows = [r for r in gold if ('DUP' in (r['cheat_types'] or '').upper()
            or 'RESPAWN' in (r['cheat_types'] or '').upper()) and r['time_range']]
    rows = rows[args.shard_index::args.num_shards][:args.limit]
    sigmas = [float(s) for s in args.sigmas.split(',')]
    blocks = [f'block{i}' for i in range(args.block_lo, args.block_hi + 1)]

    rec = {}
    for r in rows:
        vid = r['video_id']; vp = key.get(vid, {}).get('video_path', '')
        if not os.path.exists(vp):
            continue
        obj = parse_obj(r['instruction'])
        frames = P.sample_frames_like_training(vp, 93, 480, 768)
        rep = np.linspace(0, 92, T_LAT).astype(int)
        # per-frame detection: locate the frame t_app where the 2nd instance
        # first appears, plus dup_cell + prim_cell
        t_app, dup_cell, prim_cell = None, None, None
        for lt, pf in enumerate(rep):
            dets = detect_all(loc, frames[pf], obj, topk=4)
            if len(dets) >= 2:
                p0 = dets[0]
                oth = [d for d in dets[1:] if abs(d[0] - p0[0]) + abs(d[1] - p0[1]) > 40]
                if oth and lt >= 2:
                    t_app = lt; dup_cell = px_to_cell(oth[0][0], oth[0][1]); prim_cell = px_to_cell(p0[0], p0[1])
                    break
        if t_app is None:
            print(f'  {vid} no valid appearance frame', flush=True); continue
        fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        fn_norm = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(r['instruction'], max_length=512).to(device)
        for sigma in sigmas:
            store = P.forward_with_hooks(models, fn_norm, emb, sigma, args.fps, device, dtype, edm)
            for b in blocks:
                if b not in store:
                    continue
                feat = store[b].float()
                T, H, W, D = feat.shape
                if t_app >= T:
                    continue
                fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                # self_at: min self-sim over the appearance window [t_app, min(t_app+1,T-1)]
                cand = [self_sim(fn, t, dup_cell) for t in range(t_app, min(t_app + 2, T))]
                self_at = min(cand) if cand else np.nan
                # self_pre: mean self-sim of that cell before appearance (1..t_app-1)
                pre = [self_sim(fn, t, dup_cell) for t in range(1, t_app)]
                self_pre = float(np.mean(pre)) if pre else np.nan
                # floor: full-sequence self-sim over 30 random background cells
                rng = np.random.RandomState(0); fl = []
                for _ in range(30):
                    ry, rx = rng.randint(H), rng.randint(W)
                    if abs(ry - dup_cell[0]) + abs(rx - dup_cell[1]) > 4 and abs(ry - prim_cell[0]) + abs(rx - prim_cell[1]) > 4:
                        for t in range(1, T):
                            fl.append(self_sim(fn, t, (ry, rx)))
                floor = float(np.mean(fl)) if fl else np.nan
                # self_mover: full-sequence self-sim of the primary cell (legal-motion control)
                mv = [self_sim(fn, t, prim_cell) for t in range(1, T)]
                self_mover = float(np.mean(mv)) if mv else np.nan
                k = (b, sigma)
                rr = rec.setdefault(k, {'at': [], 'pre': [], 'floor': [], 'mover': []})
                for nm, val in [('at', self_at), ('pre', self_pre), ('floor', floor), ('mover', self_mover)]:
                    if val == val:
                        rr[nm].append(val)
        print(f'  {vid}: t_app={t_app} dup={dup_cell}', flush=True)

    out = {}
    for (b, s), rr in rec.items():
        at, pre, fl, mv = map(lambda x: np.array(rr[x]), ['at', 'pre', 'floor', 'mover'])
        out[f'{b}|{s}'] = {
            'self_at': float(np.median(at)), 'self_pre': float(np.median(pre)),
            'floor_static': float(np.median(fl)), 'self_mover': float(np.median(mv)),
            'gap_vs_floor': float(np.median(fl) - np.median(at)),   # >0 = dup self-sim below floor
            'drop_vs_pre': float(np.median(pre) - np.median(at)),   # >0 = dup below empty baseline
            'n': len(at),
        }
    open(args.out, 'w').write(json.dumps(out, indent=1))
    print(f'[change] shard {args.shard_index} -> {args.out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=f'{GAGI}/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--sigmas', default='0.2,0.4')
    ap.add_argument('--block-lo', type=int, default=8)
    ap.add_argument('--block-hi', type=int, default=26)
    ap.add_argument('--limit', type=int, default=12)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out', required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
