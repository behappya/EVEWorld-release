"""Hard-link the 777 original mp4+txt pairs into agibot_ewm_clean/ and verify the counts.

(*_trans.mp4 preview copies are skipped: pack_data.py indexes sorted(glob('*.mp4')), so
packing them would double the entry count and shift every index.)
"""

import argparse
import os

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
SRC = f'{GAGI}/agibot_ewm_train_final'
DST = f'{GAGI}/agibot_ewm_clean'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--dst', default=DST)
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    stems = sorted(
        f[:-4] for f in os.listdir(args.src)
        if f.endswith('.mp4') and not f.endswith('_trans.mp4')
    )
    n_new = 0
    for stem in stems:
        for ext in ('.mp4', '.txt'):
            src = os.path.join(args.src, stem + ext)
            dst = os.path.join(args.dst, stem + ext)
            if not os.path.exists(src):
                raise FileNotFoundError(src)
            if not os.path.exists(dst):
                os.link(src, dst)
                n_new += 1

    mp4 = [f for f in os.listdir(args.dst) if f.endswith('.mp4')]
    txt = [f for f in os.listdir(args.dst) if f.endswith('.txt')]
    trans = [f for f in mp4 if f.endswith('_trans.mp4')]
    assert len(mp4) == 777 and len(txt) == 777, (len(mp4), len(txt))
    assert not trans, trans[:3]
    print(f'[a1] ok: {args.dst} mp4={len(mp4)} txt={len(txt)} trans=0 (new links={n_new})')


if __name__ == '__main__':
    main()
