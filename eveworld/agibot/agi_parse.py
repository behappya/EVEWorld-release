"""AgiBot dual-arm instruction parsing: skill from task_info, entities from verb templates.

Replaces the GR1 parse_objects regex, which scores 0 hits across the 777 clips.
"""

import argparse
import json
import os
import re
from functools import lru_cache

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
TASK_INFO_DIR = f'{GAGI}/agibot_ewm_raw/task_info'
CLEAN_DIR = f'{GAGI}/agibot_ewm_clean'

TRANSFER = {'Pick', 'Place', 'HandOver', 'Insert', 'Push', 'Pull'}
STATE = {'Pour', 'Open', 'Close', 'PressButton', 'Brush', 'Shake', 'Hold'}

# arm phrases (stripped from the sentence to simplify the entity templates)
ARM_RES = [
    re.compile(r'\s*with (?:the )?(left|right|both) (?:arms?|hands?)', re.I),
    re.compile(r'\s*held in (?:the )?(left|right) (?:arm|hand)', re.I),
    re.compile(r'\s*using (?:the )?(left|right|both) (?:arms?|hands?)', re.I),
]

# verb templates tried in order; roles name the semantics of each capture group
VERB_PATTERNS = [
    (re.compile(r'^pick up (?:the )?(.+?)(?: (?:from|out of|off) (?:the )?(.+))?$'), ('object', 'source')),
    (re.compile(r'^(?:place|put) (?:the )?(.+?) (?:on|onto|in|into|inside|to|at|under|over|near|beside|next to) (?:the )?(.+)$'), ('object', 'dest')),
    (re.compile(r'^(?:empty|pour) (?:out )?(?:the )?(.+?) (?:from|out of|in|into) (?:the )?(.+)$'), ('content', 'object')),
    (re.compile(r'^(?:open|close) (?:the )?(.+)$'), ('object',)),
    (re.compile(r'^(?:hand over|hand|pass|give) (?:the )?(.+?)(?: to (?:the )?(.+))?$'), ('object', 'dest')),
    (re.compile(r'^(?:push|pull) (?:the )?(.+?)(?: (?:to|toward|towards|into|out of|from) (?:the )?(.+))?$'), ('object', 'dest')),
    (re.compile(r'^(?:hold|grasp|grab) (?:the )?(.+)$'), ('object',)),
    (re.compile(r'^press (?:the )?(.+)$'), ('object',)),
    (re.compile(r'^(?:brush|shake|wipe) (?:(?:off|out) )?(?:the )?(.+)$'), ('object',)),
    (re.compile(r'^insert (?:the )?(.+?)(?: (?:in|into) (?:the )?(.+))?$'), ('object', 'dest')),
]
FALLBACK_RE = re.compile(r'^[a-z]+(?: [a-z]+)? (?:the )?(.+)$')
STATE_PART_RE = re.compile(r"(?:^|\s|')s?\s*(lid|door|cap|cover|button|switch|drawer)\b")
HELD_GENERIC = {'item', 'object', 'it', 'held item', 'held object'}


@lru_cache(maxsize=None)
def load_task_index():
    idx = {}
    for fn in os.listdir(TASK_INFO_DIR):
        m = re.match(r'task_(\d+)\.json$', fn)
        if not m:
            continue
        task = m.group(1)
        eps = json.load(open(os.path.join(TASK_INFO_DIR, fn)))
        idx[task] = {str(ep['episode_id']): ep for ep in eps}
    return idx


def clip_meta(name):
    """'367_648961_0' -> (episode dict, action dict, si)."""
    task, ep, si = name.rsplit('_', 2)[0], *name.rsplit('_', 2)[1:]
    epd = load_task_index()[task][ep]
    act = epd['label_info']['action_config'][int(si)]
    return epd, act, int(si)


def _strip_arm(text):
    arm = None
    for r in ARM_RES:
        m = r.search(text)
        if m:
            arm = m.group(1).lower()
            text = r.sub('', text)
    return text, arm


def _clean_np(s):
    if not s:
        return None
    s = s.strip().strip('.,').strip()
    s = re.sub(r'^held ', '', s)
    s = s.replace("'s", '')
    s = re.sub(r'\s+(neatly|gently|carefully|slowly|properly|firmly)$', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s or None


def _resolve_generic(task, ep, si, obj):
    """If object is generic ('item'...), walk back through earlier actions of the
    episode for a name."""
    epd = load_task_index()[task][str(ep)]
    for j in range(si - 1, -1, -1):
        prev = epd['label_info']['action_config'][j]
        e = _parse_text(prev.get('action_text', ''))
        cand = e.get('object')
        if cand and cand not in HELD_GENERIC:
            return cand
    return obj


def _parse_text(text):
    t = text.strip().rstrip('.').lower()
    t, arm = _strip_arm(t)
    t = t.strip().rstrip('.,').strip()
    out = {'arm': arm, 'object': None, 'source': None, 'dest': None, 'content': None}
    for pat, roles in VERB_PATTERNS:
        m = pat.match(t)
        if m:
            for role, g in zip(roles, m.groups()):
                if g:
                    out[role] = _clean_np(g)
            break
    else:
        m = FALLBACK_RE.match(t)
        if m:
            out['object'] = _clean_np(m.group(1))
    return out


def parse_clip(name):
    epd, act, si = clip_meta(name)
    skill = act.get('skill') or 'Pick'
    category = 'TRANSFER' if skill in TRANSFER else 'STATE'
    e = _parse_text(act.get('action_text', ''))
    obj = e['object']
    task, ep = name.rsplit('_', 2)[0], name.rsplit('_', 2)[1]
    if obj in HELD_GENERIC or obj is None:
        obj = _resolve_generic(task, ep, si, obj)
    # an arm phrase is useless as a GDINO container query (HandOver 'to the right arm');
    # set to None -> gate=False, and the paste bias falls back to A-in-place/background
    if e['dest'] and re.search(r'\b(arm|hand)s?\b', e['dest']):
        e['dest'] = None
    state_part = None
    if category == 'STATE':
        m = STATE_PART_RE.search((act.get('action_text') or '').lower())
        part = m.group(1) if m else None
        state_part = f'{obj} {part}'.strip() if part and part not in (obj or '') else obj
    return {
        'skill': skill, 'category': category, 'arm': e['arm'],
        'object': obj, 'source': e['source'], 'dest': e['dest'],
        'state_part': state_part, 'task_name': epd.get('task_name'),
    }


def selftest():
    stems = sorted(f[:-4] for f in os.listdir(CLEAN_DIR) if f.endswith('.mp4'))
    assert len(stems) == 777
    from collections import Counter
    ok_obj, no_obj, generic = 0, [], []
    by_skill, arm_c = Counter(), Counter()
    dest_by_cat = Counter()
    for s in stems:
        r = parse_clip(s)
        by_skill[r['skill']] += 1
        arm_c[r['arm']] += 1
        if not r['object']:
            no_obj.append(s)
        elif r['object'] in HELD_GENERIC:
            generic.append((s, r['object']))
        else:
            ok_obj += 1
        if r['dest']:
            dest_by_cat[r['category']] += 1
    print(f'object parse: ok={ok_obj}/777, empty={len(no_obj)}, generic-leftover={len(generic)}')
    print(f'skill distribution: {dict(by_skill)}')
    print(f'arm distribution: {dict(arm_c)}')
    print(f'dest non-empty (by category): {dict(dest_by_cat)}')
    if no_obj:
        print('empty-object samples:')
        for s in no_obj[:8]:
            _, act, _ = clip_meta(s)
            print(f'  {s}: {act["action_text"]!r}')
    if generic:
        print('generic-leftover samples:', generic[:5])
    # show one parse result per skill
    seen = set()
    for s in stems:
        r = parse_clip(s)
        if r['skill'] in seen:
            continue
        seen.add(r['skill'])
        _, act, _ = clip_meta(s)
        print(f'  [{r["skill"]:9s}] {act["action_text"][:64]!r}\n'
              f'      -> obj={r["object"]!r} src={r["source"]!r} dest={r["dest"]!r} '
              f'state={r["state_part"]!r} arm={r["arm"]}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--name')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.name:
        print(json.dumps(parse_clip(a.name), indent=2))
