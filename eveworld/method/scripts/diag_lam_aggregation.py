#!/usr/bin/env python3
"""EVE diagnose how LAD transition_error is aggregated (root-cause the go/no-go failure).

The mean is diluted by duplicated zero-motion frames (teleport/freeze_jump can even score below
real); compare real vs corruptions under mean/max/p90/top3 to find the separating aggregation."""
import argparse, os, sys, statistics as st
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--val-frac", type=float, default=0.3)
    a = ap.parse_args()

    import torch
    from method.lam.latent_action_model import LatentActionModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.lam, map_location=dev)
    lam = LatentActionModel(latent_ch=ck["z_dim"], action_dim=ck["action_dim"], codebook=ck["codebook"]).to(dev)
    lam.load_state_dict(ck["state_dict"]); lam.eval()

    blob = torch.load(a.latents, map_location="cpu")
    lats = blob["latents"]
    n_val = max(2, int(len(lats) * a.val_frac))
    val = lats[-n_val:]        # use the tail as validation
    print(f"[diag] {len(val)} val latents, z_dim={ck['z_dim']}", flush=True)

    def corrupt(z, kind):
        B, C, T, H, W = z.shape
        out = z.clone()
        if kind == "teleport":
            k = max(1, int(T * 0.4)); out[:, :, :k] = z[:, :, -1:].expand(-1, -1, k, -1, -1)
        elif kind == "shuffle":
            out = z[:, :, torch.randperm(T)]
        elif kind == "freeze_jump":
            k = int(T * 0.55)
            out[:, :, :k] = z[:, :, :1].expand(-1, -1, k, -1, -1)
            out[:, :, k:] = z[:, :, -1:].expand(-1, -1, T - k, -1, -1)
        elif kind == "teleport_single":
            # closer to real laziness: insert a single teleport mid-way (one frame jumps to
            # the end state and back), without creating large duplicated segments
            m = T // 2
            out[:, :, m] = z[:, :, -1]
        return out

    # several aggregations: each corruption vs real, paired per video
    aggs = {
        "mean": lambda te: te.mean().item(),
        "max": lambda te: te.max().item(),
        "p90": lambda te: te.quantile(0.9).item(),
        "top3mean": lambda te: te.topk(min(3, te.numel())).values.mean().item(),
    }
    kinds = ["teleport", "shuffle", "freeze_jump", "teleport_single"]
    # collect
    data = {ag: {"real": [], **{k: [] for k in kinds}} for ag in aggs}
    with torch.no_grad():
        for zi in val:
            z = zi.to(dev).float().unsqueeze(0)
            if z.shape[2] < 4:
                continue
            te_real = lam.transition_error(z).flatten()
            for ag, fn in aggs.items():
                data[ag]["real"].append(fn(te_real))
            for k in kinds:
                te = lam.transition_error(corrupt(z, k)).flatten()
                for ag, fn in aggs.items():
                    data[ag][k].append(fn(te))

    print(f"\n{'agg':10}{'kind':16}{'val':>10}{'x_real':>9}{'%higher':>9}")
    print("-" * 55)
    for ag in aggs:
        real_m = st.mean(data[ag]["real"])
        print(f"{ag:10}{'real':16}{real_m:>10.4f}")
        for k in kinds:
            m = st.mean(data[ag][k])
            higher = sum(1 for r, c in zip(data[ag]["real"], data[ag][k]) if c > r) / max(1, len(data[ag]["real"]))
            flag = " *" if (m / (real_m + 1e-8) >= 1.3 and higher >= 0.7) else ""
            print(f"{'':10}{k:16}{m:>10.4f}{m/(real_m+1e-8):>9.2f}{higher*100:>8.0f}%{flag}")
    print("\n[criterion] pick the aggregation with x_real>=1.3 and %higher>=70 as the LAD laziness score.")


if __name__ == "__main__":
    main()
