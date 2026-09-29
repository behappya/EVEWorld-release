#!/usr/bin/env python3
"""AgiBot GDINO entity detection over the 777 clips -> agibot_t4g_probe/t4g_anno/<name>.json
(GR1-compatible schema; entity names come from agi_parse, not the parse_objects regex).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, HERE)

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
CLEAN = f'{GAGI}/agibot_ewm_clean'
OUT_DEFAULT = f'{GAGI}/eve_v2_outputs/agibot_t4g_probe/t4g_anno'

from eveworld.pipeline.annotate import detect as D  # noqa: E402
import agi_parse  # noqa: E402

D.VIDEO_ROOT = CLEAN
D.WIMG = 640
D.HIMG = 480
D.W_LAT = 40
D.H_LAT = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--out-dir', default=OUT_DEFAULT)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    from eveworld.pipeline.annotate.gdino import GDinoLocator

    names = sorted(f[:-4] for f in os.listdir(CLEAN)
                   if f.endswith('.mp4') and not f.endswith('_trans.mp4'))
    names = names[a.shard_index::a.num_shards]
    if a.limit:
        names = names[:a.limit]
    loc = GDinoLocator(device=a.device)
    print(f'[agi-detect] shard {a.shard_index}/{a.num_shards}: {len(names)} clips', flush=True)
    for name in names:
        out = f'{a.out_dir}/{name}.json'
        if os.path.exists(out):
            continue
        try:
            ents = agi_parse.parse_clip(name)
            prompt = open(f'{CLEAN}/{name}.txt').read().strip()
            # inject entity names per clip (D.process calls D.parse_objects internally)
            D.parse_objects = lambda _p, e=ents: {
                'mover': e['object'], 'src': e['source'], 'tgt': e['dest']}
            anno = D.process(loc, name, prompt)
            anno.update(skill=ents['skill'], category=ents['category'], arm=ents['arm'],
                        state_name=ents['state_part'])
            if ents['category'] == 'STATE' and ents['state_part']:
                fr0 = D.P.sample_frames_like_training(
                    f'{CLEAN}/{name}.mp4', D.NUM_FRAMES, D.HIMG, D.WIMG)[0]
                sd = D.detect_all(loc, fr0, ents['state_part'], topk=1)
                anno['state_cells'] = D.box_to_cells(sd[0][2]) if sd else []
            json.dump(anno, open(out, 'w'), ensure_ascii=False)
            print(f'  {name}: [{ents["skill"]:8s}] gate={anno["gate_enabled"]}({anno["gate_reason"]}) '
                  f't*={anno["t_arrival"]} det={anno["n_detected_frames"]}/24', flush=True)
        except Exception as e:  # noqa: BLE001
            print(f'  {name} FAIL: {str(e)[:120]}', flush=True)
    print('[agi-detect] shard done', flush=True)


if __name__ == '__main__':
    main()
