#!/usr/bin/env python3
"""T4G-MEASURE: clean measurement (self-check fixes A/B/C/D/F/G) -- empirical basis for the loss rewrite.

Per layer per sigma: relay-tracking EPE vs GDINO trajectory; obj_self/bg similarity for margin
calibration (margin must sit between the high bg range and obj_self). GPU job.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from eveworld.pipeline.probe import probe as P

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

ANNO_DIR = f'{GAGI}/eve_v2_outputs/track4gen_probe/t4g_anno'


def chained_epe_and_sim(feat, tgt_cells):
    """feat (T,H,W,D); tgt_cells: list[(gy,gx) or None] per-frame GDINO object cells.
    Returns (epe_median, obj_self_list, bg_list). Relay: query = previous object-cell feature."""
    import torch
    feat = feat.float()
    T, H, W, D = feat.shape
    fn = feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)
    epes, obj_self, bg = [], [], []
    rng = np.random.RandomState(0)
    for t in range(1, T):
        c0, c1 = tgt_cells[t - 1], tgt_cells[t]
        if c0 is None or c1 is None:
            continue
        q = fn[t - 1, c0[0], c0[1]]                        # previous-frame object feature (relay)
        sim = (fn[t].reshape(H * W, D) @ q)                # (H*W,) full-grid similarity at frame t
        amax = int(sim.argmax()); pr = (amax // W, amax % W)
        epes.append(((pr[0] - c1[0]) ** 2 + (pr[1] - c1[1]) ** 2) ** 0.5)
        obj_self.append(float(sim[c1[0] * W + c1[1]]))     # true-cell similarity (self-similarity)
        # background: 20 random cells (excluding near-object cells)
        for _ in range(20):
            ry, rx = rng.randint(H), rng.randint(W)
            if abs(ry - c1[0]) + abs(rx - c1[1]) > 4:
                bg.append(float(sim[ry * W + rx]))
    return (float(np.median(epes)) if epes else np.nan, obj_self, bg)


def run(args):
    import torch
    from giga_models.nn import EDMLoss
    dtype = torch.bfloat16
    device = 'cuda'
    edm = EDMLoss(sigma_method=3, p_mean=0.0, p_std=1.0, use_flow=True, sigma_data=1.0)
    models = P.load_models(f'{args.model_dir}/transformer', f'{args.model_dir}/vae',
                           f'{args.model_dir}/text_encoder', device, dtype)
    sigmas = [float(s) for s in args.sigmas.split(',')]
    vids = [v.strip() for v in args.vids.split(',') if v.strip()]
    vids = vids[args.shard_index::args.num_shards]
    blocks = [f'block{i}' for i in range(args.block_lo, args.block_hi + 1)]

    rec = {}   # (block,sigma) -> {epe:[], obj:[], bg:[]}
    for vid in vids:
        ap = f'{ANNO_DIR}/{vid}.json'
        if not os.path.exists(ap):
            continue
        anno = json.load(open(ap))
        tgt_cells = [f['target_cell'] for f in anno['per_lat_frame']]
        frames = P.sample_frames_like_training(
            f'{GAGI}/gr1_finetune_data/raw_data/{vid}.mp4', 93, 480, 768)
        fr = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
        frames_norm = ((fr - 0.5) / 0.5).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = models['text_encoder'].encode_prompts(anno['prompt'], max_length=512).to(device)
        for sigma in sigmas:
            store = P.forward_with_hooks(models, frames_norm, emb, sigma, args.fps, device, dtype, edm)
            for b in blocks:
                if b not in store:
                    continue
                epe, obj, bg = chained_epe_and_sim(store[b], tgt_cells)
                r = rec.setdefault((b, sigma), {'epe': [], 'obj': [], 'bg': []})
                if epe == epe:
                    r['epe'].append(epe)
                r['obj'] += obj
                r['bg'] += bg
        print(f'[measure] {vid} done', flush=True)

    out = {}
    for (b, s), r in rec.items():
        obj = np.array(r['obj']); bg = np.array(r['bg'])
        out[f'{b}|{s}'] = {
            'epe_median': float(np.median(r['epe'])) if r['epe'] else None,
            'obj_self_mean': float(obj.mean()) if len(obj) else None,
            'obj_self_p10': float(np.percentile(obj, 10)) if len(obj) else None,
            'bg_mean': float(bg.mean()) if len(bg) else None,
            'bg_p90': float(np.percentile(bg, 90)) if len(bg) else None,
            'separation': float(np.percentile(obj, 10) - np.percentile(bg, 90)) if len(obj) and len(bg) else None,
            'n': len(r['epe']),
        }
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f'[measure] shard {args.shard_index} -> {args.out}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', default=f'{GAGI}/eve_v2_outputs/anchor_models/round0_ema_st')
    ap.add_argument('--vids', required=True)
    ap.add_argument('--sigmas', default='0.2,0.4,0.7')
    ap.add_argument('--block-lo', type=int, default=8)
    ap.add_argument('--block-hi', type=int, default=26)
    ap.add_argument('--fps', type=int, default=16)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out', required=True)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
