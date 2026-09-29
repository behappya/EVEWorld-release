#!/usr/bin/env python3
"""X9: EVAL-175 official DreamGenBench generation dispatcher - one GPU per model x split unit (seed 42, 2 models x 3 splits), kjobctl-compatible."""
import argparse
import os
import subprocess
import sys
import time

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
MODELS = {
    'round0': f'{GAGI}/eve_v2_outputs/anchor_models/round0_ema_st',
    'anmix_s200': f'{GAGI}/eve_v2_outputs/anchor_models/probe_anmix_s200',
    'anmix_s50': f'{GAGI}/eve_v2_outputs/anchor_models/probe_anmix_s50',
    'dpo_b500_s50': f'{GAGI}/eve_v2_outputs/anchor_models/probe_dpo_b500_s50',
    # full set: L_id + static patch + weight map
    't4g_wmapA_s150': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_s150',
    # ablation: weight map only
    't4g_wmaponly_s150': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmaponly_s150',
    # early A stage (avoids choosing an over-memorized checkpoint)
    't4g_wmapA_s50': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_s50',
    't4g_wmapA_s100': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_s100',
    # raw pretrain base (before round0)
    'pretrain': f'{GAGI}/giga_world_0_video_pretrain',
    # official GigaWorld-0-Video-GR1-2b downloaded release
    'gr1_2b': f'{GAGI}/giga_world_0_video_gr1',
    # A recipe trained straight from pretrain (skips the fine-tune tax)
    't4g_wmapA_pre_s100': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_s100',
    't4g_wmapA_pre_s150': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_s150',
    't4g_wmapA_pre_s200': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_s200',
    't4g_wmapA_pre_s250': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_s250',
    't4g_wmapA_pre_s300': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_s300',
    't4g_wmapA_pre_cleanv4_u3_s50': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_cleanv4_u3_s50',
    't4g_wmapA_pre_cleanv4_u3_s100': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_cleanv4_u3_s100',
    't4g_wmapA_pre_cleanv4_u3_s150': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_cleanv4_u3_s150',
    't4g_wmapA_pre_cleanv4_u3_s200': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_cleanv4_u3_s200',
    't4g_wmapA_pre_cleanv4_u3_s250': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_cleanv4_u3_s250',
    't4g_wmapA_pre_cleanv4_u3_s300': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_cleanv4_u3_s300',
    't4g_wmapA_pre_seed42_s50': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_seed42_s50',
    't4g_wmapA_pre_seed42_s100': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_seed42_s100',
    't4g_wmapA_pre_seed42_s150': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_seed42_s150',
    't4g_wmapA_pre_seed42_s200': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_seed42_s200',
    't4g_wmapA_pre_seed42_s250': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_seed42_s250',
    't4g_wmapA_pre_seed42_s300': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_seed42_s300',
    't4g_wmapA_pre_noaug_s50': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_noaug_s50',
    't4g_wmapA_pre_noaug_s100': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_noaug_s100',
    't4g_wmapA_pre_noaug_s150': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_noaug_s150',
    't4g_wmapA_pre_noaug_s200': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_noaug_s200',
    't4g_wmapA_pre_noaug_s250': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_noaug_s250',
    't4g_wmapA_pre_noaug_s300': f'{GAGI}/eve_v2_outputs/anchor_models/probe_t4g_wmapA_pre_noaug_s300',
}
SPLITS = ['gr1_env', 'gr1_object', 'gr1_behavior']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-base', required=True)
    ap.add_argument('--seed', default='42')
    ap.add_argument('--gen-script', default='eveworld/method/scripts/generate_eag.py')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--lam', default=f'{GAGI}/eve_outputs/lam/lam_gr1.pt')
    ap.add_argument('--models', nargs='*', default=None, help='default all; may pass a subset, e.g. anmix_s50')
    ap.add_argument('--splits', nargs='*', default=None, help='default the three official splits; may pass e.g. gr92p0..7')
    a = ap.parse_args()

    sel = a.models or list(MODELS)
    splits = a.splits or SPLITS
    units = [(m, s) for m in sel for s in splits]
    procs = []
    for gpu, (model, split) in enumerate(units):
        md = MODELS[model]
        save = os.path.join(a.out_base, model, split)
        os.makedirs(save, exist_ok=True)
        cmd = [a.python, a.gen_script,
               '--data-path', f'{GAGI}/gr1_dreamgen_eval/giga_input/eval175_{split}.json',
               '--save-dir', save,
               '--transformer', f'{md}/transformer', '--text-encoder', f'{md}/text_encoder',
               '--vae', f'{md}/vae', '--lam', a.lam,
               '--eag-weight', '0', '--num-inference-steps', '30', '--num-frames', '93',
               '--height', '480', '--width', '768', '--fps', '16',
               '--seed', a.seed, '--limit', '0', '--skip-existing']
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
        lf = open(os.path.join(save, 'gen.log'), 'w')
        print(f'[x9] GPU {gpu} <- {model}/{split} -> {save}', flush=True)
        procs.append((model, split, subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT), save))

    fail = 0
    while True:
        alive = [p for p in procs if p[2].poll() is None]
        stat = ' '.join(f"{m[:3]}/{s.split('_')[-1][:3]}:{len([f for f in os.listdir(os.path.join(sv,'generated_only')) if f.endswith('.mp4')]) if os.path.isdir(os.path.join(sv,'generated_only')) else 0}{'[run]' if pr.poll() is None else f'[done {pr.returncode}]'}"
                        for m, s, pr, sv in procs)
        print(f'[x9] {time.strftime("%H:%M:%S")} {stat}', flush=True)
        if not alive:
            break
        time.sleep(60)
    for m, s, pr, sv in procs:
        if pr.returncode != 0:
            fail += 1
            print(f'[x9] FAIL {m}/{s} rc={pr.returncode} log={sv}/gen.log', flush=True)
    print(f'[x9] all done fail={fail}', flush=True)
    sys.exit(1 if fail else 0)


if __name__ == '__main__':
    main()
