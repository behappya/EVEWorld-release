#!/usr/bin/env python3
"""Assemble a training checkpoint (plus the pretrain base's vae/text_encoder) into a probe model
dir: agibot_ewm_apre/probes/agi_apre_<arm>_s<step>/{transformer,vae,text_encoder}.
"""
import argparse
import glob
import os
import re

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
APRE = f'{GAGI}/eve_v2_outputs/agibot_ewm_apre'          # training checkpoints live here
PROBE_ROOT = f'{GAGI}/agibot_ewm_apre/probes'           # probe location the gen harness expects
PRETRAIN = f'{GAGI}/giga_world_0_video_pretrain'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='agi_apre_wmaponly_s800',
                    help='training run name (locates experiments and the probe arm name)')
    ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--ema', action='store_true', help='use transformer_ema instead of transformer')
    a = ap.parse_args()

    pats = glob.glob(f'{APRE}/experiments/**/checkpoint_*_step_{a.step}', recursive=True)
    pats = [p for p in pats if os.path.isdir(p)]
    if not pats:
        raise SystemExit(f'no step {a.step} checkpoint found: {APRE}/experiments')
    ckpt = sorted(pats)[-1]
    tf_name = 'transformer_ema' if a.ema else 'transformer'
    tf = os.path.join(ckpt, tf_name)
    if not os.path.isfile(os.path.join(tf, 'config.json')):
        raise SystemExit(f'checkpoint missing {tf_name}/config.json: {tf}')

    arm = re.sub(r'_s\d+$', '', a.run)                      # agi_apre_wmaponly
    # the gen/eval harness only accepts the ewm_apre_* prefix (kjob_ewmbench_8gpu_serial.sh case)
    arm = re.sub(r'^agi_apre', 'ewm_apre', arm)             # ewm_apre_wmaponly
    probe_name = f'{arm}_s{a.step}'                         # ewm_apre_wmaponly_s50
    probe = f'{PROBE_ROOT}/{probe_name}'
    os.makedirs(probe, exist_ok=True)
    for name, src in (('transformer', tf),
                      ('vae', f'{PRETRAIN}/vae'),
                      ('text_encoder', f'{PRETRAIN}/text_encoder')):
        dst = os.path.join(probe, name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
    print(f'probe OK: {probe}\n  transformer -> {tf}')
    print(f'  for generation: MODELS={probe_name}')


if __name__ == '__main__':
    main()
