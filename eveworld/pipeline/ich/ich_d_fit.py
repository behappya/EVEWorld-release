#!/usr/bin/env python3
"""ICH-D frozen detector fit (B1 D-arm): full-set LR over 40 cases; weights frozen for ICHDModule.

16-dim v1 features: block10/12/16 g1p novelty + block22 CIC, order as cic_match fuse_loo B;
rows = sigma 0.2 and 0.4 merged. LOO ref 0.8261@sigma0.4. Output {w,b,mu,sd,meta}. CPU-only.
"""
import argparse
import json
import os

import numpy as np

from eveworld.pipeline.selfcase import g1 as G1

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
CIC_DIR = f'{GAGI}/eve_v2_outputs/selfcase/cic_match'
CACHE_DEFAULT = os.path.join(CIC_DIR, 'cic_cell_features.npz')
OUT_DEFAULT = os.path.join(CIC_DIR, 'ich_d_frozen_lr.npz')
A_BLOCKS = ['block10', 'block12', 'block16']
SIGMAS = [0.2, 0.4]


def build_matrix(d, sigma):
    """16-dim v1 B order: block10/12/16 novelty + block22 CIC."""
    return np.hstack([d[f'X_{b}_{sigma}'] for b in A_BLOCKS] + [d[f'X_cic_{sigma}']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=CACHE_DEFAULT)
    ap.add_argument('--out', default=OUT_DEFAULT)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    meta_in = json.loads(str(d['meta']))
    assert meta_in['cic_block'] == 'block22', meta_in
    y1 = d['y']
    Xs = {s: build_matrix(d, s) for s in SIGMAS}
    X = np.vstack([Xs[s] for s in SIGMAS])
    y = np.concatenate([y1] * len(SIGMAS))
    print(f'[ich-d fit] {meta_in["n_cases"]} cases, rows/sigma={len(y1)} '
          f'({int(y1.sum())} dup / {int((1 - y1).sum())} bg), merged rows={len(y)}, '
          f'dims={X.shape[1]}')

    mu, sd = X.mean(0), X.std(0) + 1e-8
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    clf.fit((X - mu) / sd, y)
    w = clf.coef_.ravel().astype(np.float64)
    b = clf.intercept_.astype(np.float64)

    # 对账 1: 手算 sigmoid(w·z+b) 与 sklearn predict_proba 逐位一致
    z = (X - mu) / sd
    p_manual = 1.0 / (1.0 + np.exp(-(z @ w + b[0])))
    p_sklearn = clf.predict_proba(z)[:, 1]
    assert np.allclose(p_manual, p_sklearn, atol=1e-10), '手算与 sklearn 不一致'
    print('[ich-d fit] manual sigmoid == sklearn predict_proba (atol 1e-10) OK')

    # 对账 2: 训练内 AUC + dup/bg sigmoid 分离 (固化后运行时应复现该量级)
    stats = {}
    auc_all = G1.rank_auc(p_manual[y == 1], p_manual[y == 0])
    stats['merged'] = dict(train_auc=auc_all,
                           m_dup=float(p_manual[y == 1].mean()),
                           m_bg=float(p_manual[y == 0].mean()))
    print(f'[ich-d fit] merged: train_auc={auc_all:.4f} '
          f'M_dup={stats["merged"]["m_dup"]:.4f} M_bg={stats["merged"]["m_bg"]:.4f} '
          f'sep={stats["merged"]["m_dup"] - stats["merged"]["m_bg"]:+.4f}')
    for s in SIGMAS:
        zs = (Xs[s] - mu) / sd
        p = 1.0 / (1.0 + np.exp(-(zs @ w + b[0])))
        auc = G1.rank_auc(p[y1 == 1], p[y1 == 0])
        stats[str(s)] = dict(train_auc=auc, m_dup=float(p[y1 == 1].mean()),
                             m_bg=float(p[y1 == 0].mean()))
        print(f'[ich-d fit] sigma={s}: train_auc={auc:.4f} '
              f'M_dup={stats[str(s)]["m_dup"]:.4f} M_bg={stats[str(s)]["m_bg"]:.4f} '
              f'sep={stats[str(s)]["m_dup"] - stats[str(s)]["m_bg"]:+.4f} '
              f'(LOO 参考 B_16d|{s})')

    meta = dict(cache=args.cache, sigmas=SIGMAS, feature_order=A_BLOCKS + ['cic_block22'],
                n_cases=meta_in['n_cases'], n_rows_merged=int(len(y)), stats=stats,
                loo_ref_b16d_04=0.8261)
    np.savez(args.out, w=w, b=b, mu=mu, sd=sd,
             meta=np.array(json.dumps(meta), dtype=object))
    print(f'[ich-d fit] frozen LR -> {args.out}')
    print('ICH_D_FIT_OK')


if __name__ == '__main__':
    main()
