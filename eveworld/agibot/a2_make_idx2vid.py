"""Write _packidx2vid.json (order = sorted clean stems) and md5-verify the packed set against
the clean dir entry by entry; any mismatch aborts, the dataset then has to be repacked.
"""

import argparse
import hashlib
import json
import os
import random

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
CLEAN = f'{GAGI}/agibot_ewm_clean'
PACKED = f'{GAGI}/agibot_ewm_packed'
ANNO_DIR = f'{GAGI}/eve_v2_outputs/agibot_t4g_probe/t4g_anno'


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clean', default=CLEAN)
    ap.add_argument('--packed', default=PACKED)
    ap.add_argument('--anno-dir', default=ANNO_DIR)
    ap.add_argument('--sample', type=int, default=40,
                    help='number of md5 spot checks (0 = full scan)')
    args = ap.parse_args()

    stems = sorted(
        f[:-4] for f in os.listdir(args.clean)
        if f.endswith('.mp4') and not f.endswith('_trans.mp4')
    )
    assert len(stems) == 777, len(stems)

    packed_dir = os.path.join(args.packed, 'videos', 'data')
    packed_n = len([f for f in os.listdir(packed_dir) if f.endswith('.mp4')])
    assert packed_n == 777, f'packed data_size={packed_n} != 777'

    idxs = list(range(777))
    if args.sample and args.sample < 777:
        rng = random.Random(0)
        idxs = sorted(rng.sample(idxs, args.sample))
        # always check both ends (index shifts surface at the ends first)
        for forced in (0, 776):
            if forced not in idxs:
                idxs.append(forced)
        idxs = sorted(set(idxs))
    bad = []
    for i in idxs:
        a = md5(os.path.join(packed_dir, f'{i}.mp4'))
        b = md5(os.path.join(args.clean, f'{stems[i]}.mp4'))
        if a != b:
            bad.append((i, stems[i]))
    if bad:
        raise SystemExit(
            f'[a2] md5 mismatch {len(bad)}/{len(idxs)}: {bad[:5]} -> repack from the clean dir'
        )

    os.makedirs(args.anno_dir, exist_ok=True)
    out = os.path.join(args.anno_dir, '_packidx2vid.json')
    with open(out, 'w') as f:
        json.dump({str(i): s for i, s in enumerate(stems)}, f, indent=0)
    print(f'[a2] ok: md5 checked {len(idxs)}/777, idx2vid -> {out}')


if __name__ == '__main__':
    main()
