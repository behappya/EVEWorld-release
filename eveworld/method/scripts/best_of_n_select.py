#!/usr/bin/env python3
"""EVE track B: Qwen best-of-N rejection sampling — judge A selects, independent judge B
evaluates (separation breaks the winner's-curse circularity; without --eval-csvs it degrades
to A-selects/A-evaluates, a quick look only)."""
import argparse, csv, json, os
import statistics as st
from itertools import combinations


def load_qwen(csv_path):
    """video basename -> (severity, any_shortcut)."""
    out = {}
    if not csv_path or not os.path.exists(csv_path):
        return out
    for r in csv.DictReader(open(csv_path)):
        name = os.path.basename(r.get("video_path", ""))
        try:
            sev = float(r.get("laziness_severity", 0))
        except Exception:
            sev = 0.0
        try:
            anysc = float(r.get("any_shortcut", 0))
        except Exception:
            anysc = 0.0
        out[name] = (sev, anysc)
    return out


def common_prompts(scores):
    prompts = set(scores[0].keys())
    for s in scores[1:]:
        prompts &= set(s.keys())
    return sorted(prompts)


def select_and_eval(sel, ev, prompts, N):
    """Judge A selects per prompt, judge B scores the selected set; all reductions stay within one scale."""
    rand_A, rand_B = [], []          # random-selection expectation (scale A / scale B)
    bon_ev_sev, bon_ev_any = [], []  # best-of-N (A selects) scored on scale B
    self_ev = []                     # best-of-N (A selects) scored on scale A (self-reported)
    oracle_ev = []                   # B selects, B evaluates (theoretical upper bound)
    for p in prompts:
        sel_sev = [sel[i].get(p, (99, 1))[0] for i in range(N)]
        ev_sev = [ev[i].get(p, (99, 1))[0] for i in range(N)]
        ev_any = [ev[i].get(p, (99, 1))[1] for i in range(N)]
        rand_A.append(st.mean(sel_sev)); rand_B.append(st.mean(ev_sev))
        # best-of-N: judge A picks the smallest severity (tie-break on A's any)
        best_i = min(range(N), key=lambda i: (sel_sev[i], sel[i].get(p, (99, 1))[1]))
        bon_ev_sev.append(ev_sev[best_i]); bon_ev_any.append(ev_any[best_i])
        self_ev.append(sel_sev[best_i])          # same chosen candidate, scored on scale A
        oracle_ev.append(min(ev_sev))
    rA, rB = st.mean(rand_A), st.mean(rand_B)
    selfA, evalB = st.mean(self_ev), st.mean(bon_ev_sev)
    # same-scale reductions: A (self-reported/optimistic) vs B (independent); difference = circularity gap.
    red_A = rA - selfA               # scale A: random-A -> best-of-N (A-evaluated)
    red_B = rB - evalB               # scale B: random-B -> best-of-N (B-evaluated, independent)
    return {
        "random_sev_A": round(rA, 4), "random_sev_B": round(rB, 4),
        "bestofN_sev_selfA": round(selfA, 4),     # A-selects/A-scores (self-report, optimistic)
        "bestofN_sev_evalB": round(evalB, 4),     # A-selects/B-scores (independent, main paper number)
        "bestofN_any_evalB": round(st.mean(bon_ev_any), 4),
        "oracle_sev_evalB": round(st.mean(oracle_ev), 4),
        "reduction_A": round(red_A, 4),           # self-reported reduction (same A scale)
        "reduction_B": round(red_B, 4),           # independent reduction (same B scale) = main paper conclusion
        "circularity_gap": round(red_A - red_B, 4),  # circularity inflation (same-scale difference)
    }


def scaling_curve(sel, ev, prompts, Nmax, max_combos=35):
    """Scaling curve for N=2..Nmax: mean separated evaluation over seed-subset combos (truncated to the first max_combos)."""
    curve = []
    for n in range(2, Nmax + 1):
        combos = list(combinations(range(Nmax), n))
        if len(combos) > max_combos:
            combos = combos[:max_combos]
        rb, bs, rda = [], [], []
        for c in combos:
            sub_sel = [sel[i] for i in c]
            sub_ev = [ev[i] for i in c]
            r = select_and_eval(sub_sel, sub_ev, prompts, n)
            rb.append(r["random_sev_B"]); bs.append(r["bestofN_sev_evalB"]); rda.append(r["reduction_B"])
        curve.append({"N": n, "n_combos": len(combos),
                      "random_sev_B": round(st.mean(rb), 4),
                      "bestofN_sev_evalB": round(st.mean(bs), 4),
                      "reduction_B": round(st.mean(rda), 4)})
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dirs", nargs="+", required=True, help="N candidate dirs (each generated_only)")
    ap.add_argument("--sel-csvs", nargs="+", required=True, help="judge A (selection) csvs, one-to-one with cand-dirs")
    ap.add_argument("--eval-csvs", nargs="+", default=None,
                    help="judge B (independent eval) csvs, one-to-one with cand-dirs; omitted -> degrades to A-selects/A-scores (no conclusions)")
    ap.add_argument("--seeds", nargs="+", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaling", action="store_true", help="also emit the N=2..Nmax scaling curve")
    a = ap.parse_args()
    N = len(a.cand_dirs)
    assert len(a.sel_csvs) == N, "sel-csvs count must match cand-dirs"
    seeds = a.seeds or [str(i) for i in range(N)]

    sel = [load_qwen(c) for c in a.sel_csvs]
    separated = bool(a.eval_csvs)
    ev = [load_qwen(c) for c in a.eval_csvs] if separated else sel
    if separated:
        assert len(a.eval_csvs) == N, "eval-csvs count must match cand-dirs"

    prompts = common_prompts([{**s} for s in sel] + ([{**e} for e in ev] if separated else []))
    if not prompts:
        raise SystemExit("no common prompts across candidates (filenames mismatch?). Check generated_only filenames.")

    res = select_and_eval(sel, ev, prompts, N)
    per_prompt = []
    for p in prompts:
        sel_sev = [sel[i].get(p, (99, 1))[0] for i in range(N)]
        ev_sev = [ev[i].get(p, (99, 1))[0] for i in range(N)]
        best_i = min(range(N), key=lambda i: (sel_sev[i], sel[i].get(p, (99, 1))[1]))
        per_prompt.append({"prompt": p, "best_seed": seeds[best_i],
                           "sel_severity": sel_sev, "eval_severity": ev_sev,
                           "chosen_eval_severity": ev_sev[best_i]})

    summary = {"N": N, "seeds": seeds, "n_prompts": len(prompts), "separated": separated, **res}
    summary["reduction_pct_B"] = (round(100 * res["reduction_B"] / res["random_sev_B"], 1)
                                  if res["random_sev_B"] > 0 else None)

    out = {"summary": summary, "selection": per_prompt}
    if a.scaling:
        out["scaling"] = scaling_curve(sel, ev, prompts, N)
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if separated:
        print(f"\n[main result / independent judge B] random(B)={res['random_sev_B']} -> best-of-N(A-select/B-score)={res['bestofN_sev_evalB']} "
              f"(true drop {res['reduction_B']:+.3f}, {summary['reduction_pct_B']}%); oracle upper bound={res['oracle_sev_evalB']}")
        print(f"[de-circularization] drop on the same scale: A(self)={res['reduction_A']:+.3f} vs B(independent)={res['reduction_B']:+.3f}; "
              f"circularity inflation={res['circularity_gap']:+.3f} (smaller = A-selection is not just fitting A's noise)")
        print("[criterion] reduction_B significantly > 0 (independent judge confirms the drop) = best-of-N really reduces semantic laziness, not noise fitting.")
    else:
        print(f"\n[warn] --eval-csvs omitted; only the self-reported A-selects/A-scores mode (optimistic, circular). Paper conclusions require separated evaluation.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
