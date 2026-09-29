#!/usr/bin/env python3
"""EVE P0 - orthogonality analysis: scatter process score (1 - laziness) vs physics/quality
score and report the correlation; the goal is to show a high physics score can still cheat on
process, i.e. laziness is orthogonal to physical plausibility. Output: scatter png + correlation json.
"""
import argparse, json, csv, os
import numpy as np


def load_physics(path, col):
    d = {}
    if path.endswith(".json"):
        j = json.load(open(path))
        for k, v in (j.items() if isinstance(j, dict) else []):
            d[k] = float(v if not isinstance(v, dict) else v.get(col, np.nan))
    else:
        for row in csv.DictReader(open(path)):
            key = row.get("video") or row.get("file_name") or row.get("name")
            try:
                d[key] = float(row[col])
            except Exception:
                pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", required=True)
    ap.add_argument("--physics", required=True)
    ap.add_argument("--phys-col", default="score")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()

    proc = json.load(open(a.process))["per_video"]
    # process score: 1 - lazy per video, higher = more faithful
    proc_score = {r["video"]: 1.0 - r["lazy"] for r in proc}
    phys = load_physics(a.physics, a.phys_col)

    keys = [k for k in proc_score if k in phys]
    if len(keys) < 5:
        print(f"[warn] only {len(keys)} matched; check that video naming agrees (process JSON 'video' field vs physics table key)")
    x = np.array([phys[k] for k in keys])          # physics score
    y = np.array([proc_score[k] for k in keys])    # process score

    def corr(fn):
        try:
            from scipy import stats
            r = fn(x, y)
            return float(r[0]), float(r[1])
        except Exception:
            c = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan")
            return c, float("nan")
    try:
        from scipy import stats
        pear = corr(stats.pearsonr); spear = corr(stats.spearmanr)
    except Exception:
        pear = (float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else float("nan"), float("nan")); spear = pear

    # key subset: fraction of cheating videos among the high-physics half
    if len(x):
        hi = x >= np.median(x)
        cheat_in_hi = float((y[hi] < 0.5).mean()) if hi.any() else float("nan")
    else:
        cheat_in_hi = float("nan")

    res = {"n": len(keys), "pearson_r": pear[0], "pearson_p": pear[1],
           "spearman_r": spear[0], "spearman_p": spear[1],
           "frac_cheat_among_high_physics": cheat_in_hi,
           "verdict_hint": "orthogonality holds (weak correlation)" if abs(pear[0]) < 0.4 else "correlation too strong, orthogonality doubtful - needs manual review"}
    json.dump(res, open(a.out_prefix + ".json", "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.scatter(x, y, alpha=0.5, s=18)
        plt.xlabel(f"Physics/Quality score ({a.phys_col})")
        plt.ylabel("Process faithfulness (1 - lazy)")
        plt.title(f"Orthogonality: pearson r={pear[0]:.2f}")
        plt.tight_layout(); plt.savefig(a.out_prefix + ".png", dpi=140)
        print("wrote", a.out_prefix + ".png")
    except Exception as ex:
        print("[warn] plotting skipped (matplotlib missing):", ex)


if __name__ == "__main__":
    main()
