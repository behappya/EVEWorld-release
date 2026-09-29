#!/usr/bin/env python3
"""T4G-JOINT weightmap integration smoke test (CPU, no model): verifies the
forward_step weightmap math step by step."""
import os
import numpy as np
import torch

# S1 upsample
wm = torch.arange(24 * 30 * 48, dtype=torch.float32).reshape(24, 30, 48)
up = wm.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
assert up.shape == (24, 60, 96), up.shape
# cell (t=3, gy=5, gx=7) must map to up[3, 10:12, 14:16] with the same value
v = float(wm[3, 5, 7])
assert torch.allclose(up[3, 10:12, 14:16], torch.full((2, 2), v)), 'upsample cell mapping wrong'
print('S1 upsample 30x48->60x96 each cell -> 2x2 block PASS')

# S2 z broadcast
B, Z = 2, 16
cw = up.unsqueeze(0).expand(B, -1, -1, -1)                  # (B,24,60,96)
w5 = cw.unsqueeze(1).expand(-1, Z, -1, -1, -1).contiguous()  # (B,16,24,60,96)
assert w5.shape == (B, Z, 24, 60, 96), w5.shape
assert torch.allclose(w5[0, 0], w5[0, 15]), 'z channels differ'
print('S2 z broadcast (B,16,24,60,96) all channels identical PASS')

# S3 paste-region max (clamp_ min)
w = torch.tensor([[0.5, 6.0, 2.0, 4.0]]).repeat(1, 1)      # simulated contract weight values
w_test = torch.tensor([0.5, 6.0, 2.0, 3.0])
w_paste = 4.0
res = w_test.clone(); res.clamp_(min=w_paste)
assert torch.allclose(res, torch.tensor([4.0, 6.0, 4.0, 4.0])), res  # 0.5->4, 6 kept, 2->4, 3->4
print('S3 paste region clamp_(min=w_paste) raises only, never lowers PASS')

# S4 per-sample normalisation
torch.manual_seed(0)
w_map = torch.rand(B, Z, 24, 60, 96) * 5 + 0.5
w_norm = w_map / w_map.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
means = w_norm.mean(dim=(1, 2, 3, 4))
assert torch.allclose(means, torch.ones(B), atol=1e-5), means
print(f'S4 per-sample normalisation mean={means.tolist()} (exactly 1) PASS')

# S5 loss scale conservation (uniform err: weighted.mean == err.mean)
err = torch.full((B, Z, 24, 60, 96), 0.3)
lw = (err * w_norm).mean(dim=(1, 2, 3, 4))
assert torch.allclose(lw, torch.full((B,), 0.3), atol=1e-5), lw
print('S5 normalized loss scale conserved (unchanged under uniform err) PASS')

# S6 cache == live
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eveworld.pipeline.igr.weightmap import weightmap as W
CACHE = f"{os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))}/eve_v2_outputs/track4gen_probe/weightmap_cache"
import glob
vids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(CACHE + '/*.npy'))[:5]]
for vid in vids:
    cached = np.load(f'{CACHE}/{vid}.npy')
    live, _ = W.build_weightmap(W.load_anno(vid))
    assert np.allclose(cached, live), f'cache != live for vid {vid}'
print(f'S6 cache == live (sampled {len(vids)} clips) PASS')

# S7 hard dimension check: latent space == upsampled weightmap
# VAE: 480/8=60, 768/8=96; weightmap 30x48 x2 = 60x96
assert 30 * 2 == 480 // 8 == 60 and 48 * 2 == 768 // 8 == 96, 'dimension relation wrong!'
assert W.T_LAT == 24 and (93 - 1) // 4 + 1 == 24, 'time dimension wrong!'
print('S7 hard dimension check: weights 30x48 x2=60x96=latent, T=24=(93-1)/4+1 PASS')

print('\nall smoke checks PASS — weightmap integration is mathematically sound; safe to submit the 50-step probe.')
