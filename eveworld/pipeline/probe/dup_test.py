#!/usr/bin/env python3
"""T4G-DUP-TEST: first-principles live-or-dead verdict for duplication.
sep_dup = sim(duplicate, q) - bg_p90 with q = primary-instance feature; >0 = L_cnt survives.
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
H_LAT, W_LAT, T_LAT = 30, 48, 24


def parse_obj(instr):
    m = re.search(r'pick up (?:the )?(.+?)(?: from| to|\.|$)', instr, re.I)
    return m.group(1).strip() if m else instr


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
        # dup frames: latent frames in time_range with >=2 obj instances from GDINO
        rep = np.linspace(0, 92, T_LAT).astype(int)
        dup_frames = []   # (lat_t, primary_cell, dup_cell)
        for lt, pf in enumerate(rep):
            dets = detect_all(loc, frames[pf], obj, topk=4)
            if len(dets) >= 2:
                # primary = top score; duplicate = farthest other (avoids same-object boxes)
                p0 = dets[0]
                others = [d for d in dets[1:] if abs(d[0] - p0[0]) + abs(d[1] - p0[1]) > 40]
                if others:
                    dup_frames.append((lt, px_to_cell(p0[0], p0[1]), px_to_cell(others[0][0], others[0][1])))
        if not dup_frames:
            print(f'  {vid} no two-instance frame (GDINO found no duplicate)', flush=True)
            continue
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
                fnorm = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
                for (lt, pc, dc) in dup_frames:
                    if lt >= T:
                        continue
                    q = fnorm[lt, pc[0], pc[1]]
                    sim_dup = float((fnorm[lt, dc[0], dc[1]] * q).sum())
                    sm = (fnorm[lt].reshape(H * W, D) @ q).cpu().numpy()
                    rng = np.random.RandomState(0)
                    bgs = [sm[ry * W + rx] for _ in range(30)
                           for ry, rx in [(rng.randint(H), rng.randint(W))]
                           if abs(ry - pc[0]) + abs(rx - pc[1]) > 4 and abs(ry - dc[0]) + abs(rx - dc[1]) > 4]
                    k = (b, sigma)
                    rr = rec.setdefault(k, {'dup': [], 'bg': []})
                    rr['dup'].append(sim_dup); rr['bg'] += bgs
        print(f'  {vid}: {len(dup_frames)} dup frames', flush=True)

    out = {}
    for (b, s), rr in rec.items():
        dup = np.array(rr['dup']); bg = np.array(rr['bg'])
        out[f'{b}|{s}'] = {
            'sim_dup_mean': float(dup.mean()), 'sim_dup_p50': float(np.median(dup)),
            'bg_p90': float(np.percentile(bg, 90)), 'bg_mean': float(bg.mean()),
            'sep_dup': float(np.median(dup) - np.percentile(bg, 90)),  # >0 = duplicate exposed
            'n_dup': len(dup),
        }
    open(args.out, 'w').write(json.dumps(out, indent=1))
    print(f'[dup] shard {args.shard_index} -> {args.out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=f'{GAGI}/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--sigmas', default='0.2,0.4')
    ap.add_argument('--block-lo', type=int, default=8)
    ap.add_argument('--block-hi', type=int, default=26)
    ap.add_argument('--limit', type=int, default=20)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out', required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
