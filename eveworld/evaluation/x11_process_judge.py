#!/usr/bin/env python3
"""X11: four-dimension process-audit judge (D1 object permanence, D2 transport chain/teleport, D3 termination, D4 wrong actor).

Per-dimension kappa vs human annotations; only dims with kappa >= 0.4 enter the main table.
"""
import argparse
import os
import base64
import csv
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

KP = Path(os.environ.get('EVEWORLD_KAPPA_PACK', 'kappa_pack'))  # human-annotation pack, not part of this release
_TLS = threading.local()

SCHEMA = {
    "type": "object",
    "properties": {
        "target_object": {"type": "string"},
        "d1_count_first_frame": {"type": "integer", "minimum": 0, "maximum": 9},
        "d1_max_simultaneous_count": {"type": "integer", "minimum": 0, "maximum": 9},
        "d1_count_last_frame": {"type": "integer", "minimum": 0, "maximum": 9},
        "d1_reappear_at_origin_after_pick": {"type": "boolean"},
        "d1_evidence": {"type": "string", "maxLength": 200},
        "d2_transit_visible": {"type": "boolean"},
        "d2_teleport_to_goal": {"type": "boolean"},
        "d2_first_at_goal_frame": {"type": ["integer", "null"]},
        "d2_evidence": {"type": "string", "maxLength": 200},
        "d3_completion_frame": {"type": ["integer", "null"]},
        "d3_new_action_after_completion": {"type": "boolean"},
        "d3_object_moves_after_completion": {"type": "boolean"},
        "d3_settled_at_end": {"type": "boolean"},
        "d3_evidence": {"type": "string", "maxLength": 200},
        "d4_human_hand_does_task": {"type": "boolean"},
        "d4_evidence": {"type": "string", "maxLength": 150},
    },
    "required": ["target_object", "d1_count_first_frame", "d1_max_simultaneous_count",
                 "d1_count_last_frame", "d1_reappear_at_origin_after_pick", "d1_evidence",
                 "d2_transit_visible", "d2_teleport_to_goal", "d2_first_at_goal_frame", "d2_evidence",
                 "d3_completion_frame", "d3_new_action_after_completion",
                 "d3_object_moves_after_completion", "d3_settled_at_end", "d3_evidence",
                 "d4_human_hand_does_task", "d4_evidence"],
    "additionalProperties": False,
}

PROMPT = """You audit a robot-manipulation video shown as a grid of 12 frames (frame numbers burned at each tile's top-left; read left-to-right, top-to-bottom in time order). The task instruction: "{instr}"

Answer ONLY by careful visual counting and frame-by-frame comparison. Do not judge overall quality.

First identify the TARGET OBJECT being manipulated per the instruction (report as target_object).

D1 OBJECT PERMANENCE (count, do not judge):
- d1_count_first_frame: in the FIRST tile, how many separate instances of the target object are visible? (normally 1)
- d1_max_simultaneous_count: scan EVERY tile; the maximum number of instances of the target object visible SIMULTANEOUSLY in any single tile. Count carefully: an object at the source location AND another at the goal location in the same tile = 2.
- d1_count_last_frame: instances visible in the LAST tile.
- d1_reappear_at_origin_after_pick: after the object leaves its original location, does an identical object appear back at that original location in a later tile?
- d1_evidence: cite tile frame numbers for any count > first-frame count.

D2 TRANSIT FIDELITY:
- d2_transit_visible: is there a sequence of tiles showing the object HELD by the robot hand while its position changes (true transport)?
- d2_teleport_to_goal: does the object appear at/near the goal without any tile showing it being carried there?
- d2_first_at_goal_frame: frame number of the first tile where the object is at the goal (null if never).

D3 TERMINATION BEHAVIOR (critical: distinguish retreat from new action):
- d3_completion_frame: frame number where the task is DONE (object placed at goal AND hand released). null if never completed.
- d3_new_action_after_completion: after completion, does the robot START A NEW ACTION (reach toward other objects, re-grasp the placed object, aimless large motions)? Arm RETREATING to rest is NOT a new action.
- d3_object_moves_after_completion: after completion, does the placed object move again (pushed, taken back, slides)?
- d3_settled_at_end: in the final 2-3 tiles, is the scene settled (object at rest at goal; arms retreating, resting, or hovering with only tiny drift)?

D4 ACTOR:
- d4_human_hand_does_task: does a HUMAN hand/arm perform any part of the manipulation task itself (not background presence)?

Answer with JSON only."""


def client(base):
    if not hasattr(_TLS, 'c'):
        from openai import OpenAI
        _TLS.c = OpenAI(base_url=base, api_key='EMPTY')
    return _TLS.c


def _to_url(im, max_w=1600):
    if im.shape[1] > max_w:
        sc = max_w / im.shape[1]
        im = cv2.resize(im, (max_w, int(im.shape[0] * sc)))
    ok, enc = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return 'data:image/jpeg;base64,' + base64.b64encode(enc.tobytes()).decode()


def ask(base, model, img_path_or_arr, instr, timeout=300.0):
    # img_path_or_arr: path / single image array = grid mode; list[array] = per-frame
    # multi-image mode (full resolution)
    if isinstance(img_path_or_arr, list):
        content = [{"type": "text", "text": PROMPT.format(instr=instr).replace(
            'shown as a grid of 12 frames', 'shown as 12 SEPARATE full-resolution frames in time order')}]
        content += [{"type": "image_url", "image_url": {"url": _to_url(f, 1024)}} for f in img_path_or_arr]
        msgs = [{"role": "user", "content": content}]
    else:
        im = cv2.imread(str(img_path_or_arr)) if isinstance(img_path_or_arr, (str, Path)) else img_path_or_arr
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(instr=instr)},
            {"type": "image_url", "image_url": {"url": _to_url(im)}}]}]
    for attempt in range(3):
        try:
            r = client(base).chat.completions.create(
                model=model, temperature=0.0, max_tokens=1400, timeout=timeout,
                messages=msgs,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "audit", "schema": SCHEMA}},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            m = re.search(r'\{.*\}', r.choices[0].message.content or '', re.S)
            return json.loads(m.group(0)) if m else {'_error': 'no json'}
        except Exception as e:
            if attempt == 2:
                return {'_error': str(e)[:200]}
    return {'_error': 'unreachable'}


def derive_flags(p):
    """New metrics -> binary predictions for the four cheat types."""
    if p.get('_error'):
        return None
    return {
        'DUP': int(p['d1_max_simultaneous_count'] > max(1, p['d1_count_first_frame'])),
        'RESPAWN': int(p['d1_reappear_at_origin_after_pick']),
        'TELEPORT': int(p['d2_teleport_to_goal'] and not p['d2_transit_visible']),
        'WRONG_ACTOR': int(p['d4_human_hand_does_task']),
        'NO_STOP': int(p['d3_completion_frame'] is not None and
                       (p['d3_new_action_after_completion'] or p['d3_object_moves_after_completion']
                        or not p['d3_settled_at_end'])),
    }


def cohen_kappa(y1, y2):
    y1, y2 = np.array(y1), np.array(y2)
    po = (y1 == y2).mean()
    p1, p2 = y1.mean(), y2.mean()
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['calibrate', 'score'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--video-dir', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--qwen-base', default='http://127.0.0.1:8000/v1')
    ap.add_argument('--concurrency', type=int, default=16)
    ap.add_argument('--multi-image', action='store_true')
    a = ap.parse_args()

    from openai import OpenAI
    model = OpenAI(base_url=a.qwen_base, api_key='EMPTY').models.list().data[0].id
    print(f'model={model}')

    if a.mode == 'calibrate':
        rows = list(csv.DictReader(open(KP / 'gold_labels_v1.csv')))
        key = {r['video_id']: r['video_path'] for r in
               csv.DictReader(open(KP / 'hidden_key_DO_NOT_SHOW_ANNOTATORS.csv'))}
        if a.limit:
            rows = rows[:a.limit]

        def frames12(vp):
            cap = cv2.VideoCapture(str(vp))
            n = int(cap.get(7))
            idxs = np.linspace(0, n - 1, 12).astype(int)
            out, pos = [], -1
            want = set(idxs.tolist())
            while pos < idxs[-1]:
                if not cap.grab():
                    break
                pos += 1
                if pos in want:
                    ok, f = cap.retrieve()
                    if ok:
                        cv2.putText(f, f'f{pos}', (6, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                        out.append(f)
            cap.release()
            return out

        def run(r):
            if a.multi_image and key.get(r['video_id']) and Path(key[r['video_id']]).exists():
                inp = frames12(key[r['video_id']])
            else:
                inp = KP / r['grid_png']
            p = ask(a.qwen_base, model, inp, r['instruction'])
            return {'video_id': r['video_id'], 'pred': p, 'flags': derive_flags(p),
                    'gold': r['cheat_types']}
        with ThreadPoolExecutor(a.concurrency) as pool:
            results = list(pool.map(run, rows))
        with open(a.out, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        ok = [r for r in results if r['flags']]
        print(f'valid scores {len(ok)}/{len(results)}')
        print('\n=== per-dim kappa vs human gold (kappa>=0.4 enters main table) ===')
        for t in ['DUP', 'RESPAWN', 'TELEPORT', 'WRONG_ACTOR']:
            gold = [1 if t in (r['gold'] or '') else 0 for r in ok]
            pred = [r['flags'][t] for r in ok]
            k = cohen_kappa(gold, pred)
            tp = sum(g and p for g, p in zip(gold, pred))
            print(f'{t:12s} kappa={k:+.3f} gold_pos={sum(gold):3d} '
                  f'pred_pos={sum(pred):3d} hits={tp}')
        print('(NO_STOP has no human gold labels, reporting prediction rate only: '
              f"{np.mean([r['flags']['NO_STOP'] for r in ok]):.2f})")
    else:
        vids = sorted(Path(a.video_dir).glob('*.mp4'))
        if a.limit:
            vids = vids[:a.limit]

        def grid12(vp):
            cap = cv2.VideoCapture(str(vp))
            n = int(cap.get(7))
            idxs = np.linspace(0, n - 1, 12).astype(int)
            fr, pos = {}, -1
            while pos < idxs[-1]:
                if not cap.grab():
                    break
                pos += 1
                if pos in set(idxs.tolist()):
                    ok, f = cap.retrieve()
                    if ok:
                        f = cv2.resize(f, (392, 245))
                        cv2.putText(f, f'f{pos}', (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        fr[pos] = f
            cap.release()
            ts = [fr[i] for i in idxs if i in fr]
            rows_im = [np.hstack(ts[i * 4:(i + 1) * 4]) for i in range(len(ts) // 4)]
            return np.vstack(rows_im)

        def instr_of(vp):
            m = re.match(r'\d+_(.+)', vp.stem)
            return (m.group(1) if m else vp.stem).replace('_', ' ')

        def run(vp):
            p = ask(a.qwen_base, model, grid12(vp), instr_of(vp))
            return {'video': vp.name, 'pred': p, 'flags': derive_flags(p)}
        with ThreadPoolExecutor(a.concurrency) as pool:
            results = list(pool.map(run, vids))
        with open(a.out, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        ok = [r for r in results if r['flags']]
        print(f'scored {len(ok)}/{len(results)} -> {a.out}')
        for t in ['DUP', 'RESPAWN', 'TELEPORT', 'WRONG_ACTOR', 'NO_STOP']:
            print(f'{t:12s} positive_rate {np.mean([r["flags"][t] for r in ok]):.2f}')


if __name__ == '__main__':
    main()
