#!/usr/bin/env python3
"""EVE method: causal process fidelity losses (method §5, cluster-side torch): three swappable terms — margin-ranking CPC (real < cheating process), LAM dynamics consistency, monotone progress + contact gating; denoise errors come in via denoise_error_fn(z, cond)."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from .lazy_negatives import make_negatives


def cpc_loss(denoise_error_fn, z, cond, neg_kinds, margin=0.1):
    """Margin ranking: the cheating process must have a denoising error at least margin above the real one."""
    e_pos = denoise_error_fn(z, cond)                      # (B,)
    negs = make_negatives(z, neg_kinds)
    loss = z.new_zeros(())
    logs = {}
    for k, zneg in negs.items():
        e_neg = denoise_error_fn(zneg, cond)
        l = F.relu(margin - (e_neg - e_pos)).mean()
        loss = loss + l
        logs[f"cpc/{k}"] = float(l.detach())
    return loss / max(1, len(negs)), logs


def dyn_consistency_loss(lam, z):
    """LAM forward reconstruction error: sum_t ||z_{t+1} - forward(z_t, inv(z_t,z_{t+1}))||.
    Self-supervised on real video, anti-shortcut regularizer on generated video:
    lazy transitions score high."""
    B, C, T, H, W = z.shape
    zt, ztp = z[:, :, :-1], z[:, :, 1:]
    a = lam.inverse(zt, ztp)              # latent action
    pred = lam.forward_dyn(zt, a)         # predicted next frame
    return F.mse_loss(pred, ztp)


def progress_loss(progress_head, z, contact_bin=None):
    """Monotone progress + contact gating. progress_head(z)->(B,T) in [0,1]."""
    B, C, T, H, W = z.shape
    p = progress_head(z)                                   # (B,T)
    # monotonicity: penalize decreases
    mono = F.relu(p[:, :-1] - p[:, 1:]).mean()
    loss = mono
    logs = {"prog/mono": float(mono.detach())}
    # contact gating: progress before contact should stay low
    if contact_bin is not None:
        gate = torch.zeros_like(p)
        for b in range(B):
            cb = int(contact_bin[b] * (T - 1))
            gate[b, :cb] = 1.0
        pre = F.relu(p - 0.3) * gate      # penalize p>0.3 before contact
        pg = pre.mean()
        loss = loss + pg
        logs["prog/gate"] = float(pg.detach())
    return loss, logs


def total_loss(cfg, denoise_error_fn, z, cond, lam=None, progress_head=None, contact_bin=None):
    """Combine terms with cfg weights. cfg: dict(w_cpc,w_dyn,w_prog,neg_kinds,margin)."""
    total = z.new_zeros(()); logs = {}
    if cfg.get("w_cpc", 0) > 0:
        l, lg = cpc_loss(denoise_error_fn, z, cond, cfg.get("neg_kinds", ["teleport", "excision", "freeze_jump"]), cfg.get("margin", 0.1))
        total = total + cfg["w_cpc"] * l; logs.update(lg)
    if cfg.get("w_dyn", 0) > 0 and lam is not None:
        l = dyn_consistency_loss(lam, z); total = total + cfg["w_dyn"] * l; logs["dyn"] = float(l.detach())
    if cfg.get("w_prog", 0) > 0 and progress_head is not None:
        l, lg = progress_loss(progress_head, z, contact_bin); total = total + cfg["w_prog"] * l; logs.update(lg)
    logs["loss_causal_total"] = float(total.detach())
    return total, logs
