#!/usr/bin/env python3
"""EVE LAD latent-action dynamics model, self-supervised pretraining (plan 27 §4 I1): trains inverse/forward pairs on the offline Wan VAE latent cache with L = ||z_{t+1} - forward(z_t, inverse(z_t, z_{t+1}))||, no action labels; built-in go/no-go validation that transition_error separates real from synthetic lazy transitions (teleport/shuffle/freeze_jump). GPU preferred, CPU works (latents cached)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", required=True, help=".pt produced by encode_latents.py")
    ap.add_argument("--out", default="lam_pretrained.pt")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--window", type=int, default=12, help="consecutive frames per sample")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--action-dim", type=int, default=32)
    ap.add_argument("--codebook", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15, help="held-out validation fraction")
    a = ap.parse_args()

    import torch, torch.nn.functional as F
    from method.lam.latent_action_model import LatentActionModel

    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[lam] device={dev}", flush=True)

    blob = torch.load(a.latents, map_location="cpu")
    lats = blob["latents"]                       # list of (C,T,H,W)
    z_dim = blob.get("z_dim", lats[0].shape[0])
    print(f"[lam] {len(lats)} latent seqs, z_dim={z_dim}, example={tuple(lats[0].shape)}", flush=True)

    # train/val split (held-out videos; prevents LAD from merely memorizing train transitions)
    n_val = max(1, int(len(lats) * a.val_frac))
    val_lats = lats[:n_val]
    train_lats = lats[n_val:]
    print(f"[lam] train={len(train_lats)} val={len(val_lats)}", flush=True)

    def sample_batch(pool, bs, win):
        xs = []
        for _ in range(bs):
            zi = pool[torch.randint(len(pool), (1,)).item()]   # (C,T,H,W)
            T = zi.shape[1]
            if T <= win:
                seq = zi
                if T < win:  # pad by repeat last
                    seq = torch.cat([seq, seq[:, -1:].expand(-1, win - T, -1, -1)], dim=1)
            else:
                s = torch.randint(0, T - win + 1, (1,)).item()
                seq = zi[:, s:s + win]
            xs.append(seq)
        return torch.stack(xs).to(dev).float()                 # (B,C,win,H,W)

    lam = LatentActionModel(latent_ch=z_dim, action_dim=a.action_dim, codebook=a.codebook).to(dev)
    print(f"[lam] params={sum(p.numel() for p in lam.parameters())/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(lam.parameters(), lr=a.lr)

    lam.train()
    for step in range(a.steps):
        z = sample_batch(train_lats, a.batch, a.window)
        zt, ztp = z[:, :, :-1], z[:, :, 1:]
        pred = lam.forward_dyn(zt, lam.inverse(zt, ztp))
        loss = F.mse_loss(pred, ztp)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"[lam] step {step} loss {loss.item():.5f}", flush=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save({"state_dict": lam.state_dict(), "z_dim": z_dim,
                "action_dim": a.action_dim, "codebook": a.codebook}, a.out)
    print(f"[lam] saved -> {a.out}", flush=True)

    # go/no-go: transition_error separates real vs lazy
    print("\n[lam] === go/no-go: transition_error on held-out val ===", flush=True)
    lam.eval()

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
        return out

    with torch.no_grad():
        # use whole val videos, one by one (avoids mixed lengths)
        # aggregate with MAX, not mean: laziness = a few huge illegal jumps (the jump at a
        # copy-segment boundary); the mean is diluted by many zero-motion duplicated frames
        # (measured: under mean, teleport/freeze_jump score < real; max separates all).
        te = {"real": [], "teleport": [], "shuffle": [], "freeze_jump": []}
        for zi in val_lats:
            z = zi.to(dev).float().unsqueeze(0)          # (1,C,T,H,W)
            if z.shape[2] < 4:
                continue
            te["real"].append(lam.transition_error(z).max().item())
            for k in ["teleport", "shuffle", "freeze_jump"]:
                te[k].append(lam.transition_error(corrupt(z, k)).max().item())
        import statistics as st
        real_m = st.mean(te["real"])
        print(f"  real (max-agg)  = {real_m:.4f}", flush=True)
        ok = True
        for k in ["teleport", "shuffle", "freeze_jump"]:
            m = st.mean(te[k])
            ratio = m / (real_m + 1e-8)
            higher = sum(1 for r, c in zip(te["real"], te[k]) if c > r) / max(1, len(te["real"]))
            print(f"  {k:12} = {m:.4f}  (x{ratio:.2f} real, {higher*100:.0f}% higher)", flush=True)
            if ratio < 1.3 or higher < 0.7:
                ok = False
        print(f"\n[lam] go/no-go: {'PASS (lazy transfers max transition_error >> real, LAD usable for EAG)' if ok else 'FAIL (insufficient separation; tune window/steps/architecture)'}", flush=True)


if __name__ == "__main__":
    main()
