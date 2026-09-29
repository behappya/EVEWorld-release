#!/usr/bin/env python3
"""D-arm inference wrapper: restore ICH detect+erase hooks in the standard generation pipeline.

Bare from_pretrained drops ich_d.* (unexpected keys) and ICH never runs. Monkeypatches
EAGGigaWorld0Pipeline.from_pretrained; other argv passes through to generate_eag.main().
"""
import importlib.util
import os
import sys

REPO = os.environ.get('REPO_DIR', 'third_party/giga-world-0')
GIGA = os.path.dirname(REPO)
for p in (REPO, GIGA):
    if p not in sys.path:
        sys.path.insert(0, p)


def enable_ich_d_inference(transformer, transformer_dir, log_every=30):
    """Attach ICHDModule + inference hooks from transformer_dir; returns the module.

    Gates (env, default off): T4G_ICH_SIGMA_GATE='0.2,0.5' limits to in-band sigma
    rounds; T4G_ICH_M_GATE=0.7 applies Delta only where M>0.7."""
    import torch
    from giga_models.utils import load_state_dict
    from eveworld.pipeline.ich.ich_d_trainer import (ICHDModule, NOVELTY_BLOCKS,
                                                      CIC_BLOCK)

    sd = load_state_dict(transformer_dir)
    ich_sd = {k[len('ich_d.'):]: v for k, v in sd.items() if k.startswith('ich_d.')}
    del sd
    assert ich_sd, f'checkpoint 无 ich_d.* 权重 (不是 D 臂档?): {transformer_dir}'
    ich = ICHDModule(channels=transformer.config.model_channels)
    ich.load_state_dict(ich_sd, strict=True)             # strict: 缺键/多键即抛
    ref = next(transformer.parameters())
    ich.to(device=ref.device, dtype=ref.dtype)
    for name in ('lr_w', 'lr_b', 'lr_mu', 'lr_sd'):      # 固化打分保持 fp32 (同训练)
        getattr(ich, name).data = getattr(ich, name).data.float()
    ich.eval()
    ich.requires_grad_(False)
    transformer.ich_d = ich

    sigma_gate_s = os.environ.get('T4G_ICH_SIGMA_GATE', '').strip()
    m_gate = float(os.environ.get('T4G_ICH_M_GATE', '0') or 0)
    band = tuple(float(x) for x in sigma_gate_s.split(',')) if sigma_gate_s else None
    ich.set_gates(sigma_band=band, m_gate=m_gate)

    def _sigma_pre_hook(_m, _args, kwargs):
        ts = kwargs.get('timesteps')
        if ts is not None:                                # 末帧恒非 ref: c=σ/(1+σ)
            c = float(ts.reshape(ts.shape[0], -1)[0, -1])
            ich.set_sigma(c / max(1e-6, 1.0 - c))
        return None

    transformer.register_forward_pre_hook(_sigma_pre_hook, with_kwargs=True)

    state = dict(calls=0, active=0)
    for name in NOVELTY_BLOCKS:
        transformer.blocks[name].register_forward_hook(
            lambda _m, _i, out, _n=name: ich.observe(_n, out))

    def _suppress_hook(_m, _i, out):
        new = ich.suppress(CIC_BLOCK, out)
        state['calls'] += 1
        if ich.last_m is not None:
            state['active'] += 1
        if state['calls'] % log_every == 1:              # ~每条视频 1-2 行活跃证据
            m = ich.last_m
            if m is None:
                print(f'[ichd-infer] call={state["calls"]} GATED-OFF '
                      f'(sigma={ich.cur_sigma}) active={state["active"]}/{state["calls"]}',
                      flush=True)
            else:
                print(f'[ichd-infer] call={state["calls"]} M mean={m.mean().item():.4f} '
                      f'p95={m.flatten().quantile(0.95).item():.4f} '
                      f'max={m.max().item():.4f} sigma={ich.cur_sigma} '
                      f'active={state["active"]}/{state["calls"]}', flush=True)
        return new

    transformer.blocks[CIC_BLOCK].register_forward_hook(_suppress_hook)
    gate = torch.sigmoid(ich.gate.detach().float()).item()
    wout = ich.w_out.weight.detach().float().norm().item()
    print(f'[ichd-infer] ICH-D ENABLED <- {transformer_dir} | gate={gate:.4f} '
          f'wout_norm={wout:.4f} lr_w_sum={ich.lr_w.sum().item():.4f} '
          f'| observe={list(NOVELTY_BLOCKS)} suppress={CIC_BLOCK} '
          f'| sigma_gate={band} m_gate={m_gate}', flush=True)
    return ich


def main():
    spec = importlib.util.spec_from_file_location(
        'generate_eag', os.path.join(REPO, 'eveworld/method/scripts/generate_eag.py'))
    ge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ge)

    from eveworld.method import pipeline_eag
    orig = pipeline_eag.EAGGigaWorld0Pipeline.from_pretrained

    def patched_from_pretrained(*args, **kw):
        pipe = orig(*args, **kw)
        enable_ich_d_inference(pipe.transformer, kw['transformer_model_path'])
        return pipe

    pipeline_eag.EAGGigaWorld0Pipeline.from_pretrained = patched_from_pretrained
    print('[ichd-infer] from_pretrained patched; handing off to generate_eag.main()',
          flush=True)
    ge.main()


if __name__ == '__main__':
    main()
