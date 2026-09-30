#!/usr/bin/env python3
"""EVE · Track4Gen probe CPU smoke test (dev box, no GPU).

NATTEN only exists on CUDA Hopper/Blackwell (see neighborhood_attn.py), so the smoke model
falls back to torch attention via natten_parameters=None.
Run: $HOME/miniconda/envs/gigaworld/bin/python eveworld/pipeline/probe/probe_cpu_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import torch

from eveworld.pipeline.probe import probe as T

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
VIDEO = os.environ.get('SMOKE_VIDEO', f'{GAGI}/gr1_finetune_data/raw_data/13.mp4')
NUM_BLOCKS = 28


def build_tiny_model():
    """Minimal random-weight model: 28 blocks, small channels, torch attention (no NATTEN/MoE)."""
    from giga_models import GigaWorld0Transformer3DModel
    model = GigaWorld0Transformer3DModel(
        max_img_h=16, max_img_w=16, max_frames=8,
        in_channels=17, out_channels=16,
        patch_spatial=2, patch_temporal=1,
        concat_padding_mask=True,
        block_config='FA-CA-MLP',
        model_channels=64, num_blocks=NUM_BLOCKS, num_heads=4,
        mlp_ratio=2.0, crossattn_emb_channels=64, adaln_lora_dim=32,
        natten_parameters=None,   # <- key: no NATTEN, runs on CPU
        moe_parameters=None,
    )
    model.eval()
    return model


def test_hook_shapes():
    print('\n[1] hooks capture 28 blocks + shape=(B,T,H,W,D) ...')
    model = build_tiny_model()
    B, Tlat, Hlat, Wlat = 1, 6, 8, 8
    x = torch.randn(B, 17, Tlat, Hlat, Wlat)                 # (B, in_ch, T, h, w)
    timesteps = torch.rand(B, 1, Tlat, 1, 1)                 # flatten -> B*T
    crossattn = torch.randn(B, 16, 64)                       # (B, L, ctx_dim)
    padding_mask = torch.zeros(B, 1, Hlat, Wlat)

    shapes = {}
    handles = []
    for name, block in model.blocks.items():
        handles.append(block.register_forward_hook(
            lambda _m, _i, out, n=name: shapes.__setitem__(n, tuple(out.shape))))
    with torch.no_grad():
        model(x=x, timesteps=timesteps, crossattn_emb=crossattn, padding_mask=padding_mask, fps=16)
    for h in handles:
        h.remove()

    assert len(shapes) == NUM_BLOCKS, f'expected {NUM_BLOCKS} blocks, captured {len(shapes)}'
    exp = (B, Tlat, Hlat // 2, Wlat // 2, 64)                # patch_spatial=2 -> H/2, W/2; D=model_channels
    for n, s in shapes.items():
        assert len(s) == 5, f'{n} output not 5-D: {s}'
        assert s == exp, f'{n} shape={s} != expected {exp}'
    print(f'    OK: {len(shapes)} blocks, each output (B,T,H,W,D)={exp}  '
          f'(feature grid T={Tlat},H={Hlat//2},W={Wlat//2},D=64)')
    # return one real (random-weight) feature for the analysis-core test
    feat = None
    handles = []
    store = {}
    for name, block in model.blocks.items():
        handles.append(block.register_forward_hook(
            lambda _m, _i, out, n=name: store.__setitem__(n, out.detach()[0])))
    with torch.no_grad():
        model(x=x, timesteps=timesteps, crossattn_emb=crossattn, padding_mask=padding_mask, fps=16)
    for h in handles:
        h.remove()
    return store['block13']   # (T, H, W, D)


def test_analysis_core_real_frames(feat):
    print('\n[2] analysis core runs on real video frames + random features ...')
    T_lat, H, W, D = feat.shape
    n_pix = 25
    frames = T.sample_frames_like_training(VIDEO, num_frames=n_pix, height=240, width=384)
    print(f'    real frames: {frames.shape} <- {os.path.basename(VIDEO)}')
    flows = T.dense_flow_sequence(frames)
    tm = T.motion_grid(flows, H, W)
    dyn, static = T.classify_dyn_static(tm, T.DEFAULT_P_HI, T.DEFAULT_P_LO)
    print(f'    flow grid ({H}x{W})  dyn={len(dyn)}  static={len(static)}')
    m = T.compute_layer_metrics(feat, flows, dyn, static, n_pix, device='cpu')
    print(f'    metrics (random features): median_epe={m["median_epe"]:.3f} cell  '
          f'static_stability={m["static_stability"]:.3f}')
    assert np.isfinite(m['median_epe']), 'EPE not finite'
    assert np.isfinite(m['static_stability']), 'STAB not finite'
    print('    OK: analysis core runs, metrics finite')
    return frames, flows, dyn, static


def test_correctness_static(frames, flows):
    print('\n[3] correctness hard check: temporally constant features -> '
          'stability~1.0, track disp~0 ...')
    T_lat, H, W, D = 6, 4, 4, 64
    base = torch.randn(1, H, W, D)
    feat_const = base.expand(T_lat, H, W, D).contiguous()    # all frames identical
    tm = T.motion_grid(flows, H, W)
    dyn, static = T.classify_dyn_static(tm, T.DEFAULT_P_HI, T.DEFAULT_P_LO)
    if len(static) == 0:
        static = [(0, 0)]
    if len(dyn) == 0:
        dyn = [(H // 2, W // 2)]

    stab, per = T.static_stability(feat_const, static, device='cpu')
    print(f'    static stability (constant feat) = {stab:.6f}  (expect ~1.0)')
    assert stab > 0.999, f'static stability should be ~1.0, got {stab}'

    pred = T.soft_argmax_track(feat_const, dyn, device='cpu')  # (T, N, 2)
    disp = np.sqrt(((pred[1:] - pred[0:1]) ** 2).sum(-1))      # displacement relative to frame 0
    med_disp = float(np.median(disp))
    print(f'    tracked pred displacement (constant feat) median = {med_disp:.6f} cell  (expect ~0)')
    assert med_disp < 1e-3, f'constant-feature tracking should stay put, got disp {med_disp}'
    print('    OK: stability ~1.0 and track disp ~0, analysis core correct')


def main():
    print('=' * 78)
    print(' Track4Gen probe CPU smoke test (no GPU)')
    print(f'   torch={torch.__version__}  cuda_available={torch.cuda.is_available()}')
    print('=' * 78)
    feat = test_hook_shapes()
    frames, flows, dyn, static = test_analysis_core_real_frames(feat)
    test_correctness_static(frames, flows)
    print('\n' + '=' * 78)
    print(' CPU smoke: all passed (3/3).')
    print('=' * 78)


if __name__ == '__main__':
    main()
