#!/usr/bin/env python3
"""EVE method: lazy negative construction (the core of CPC, method §5-C): deterministically builds process-cheating counterparts z^- of clean latent sequences (B,C,T,H,W) without ground-truth actions (pure torch, no external model deps)."""
from __future__ import annotations
import torch


def teleport(z: "torch.Tensor", frac: float = 0.4) -> "torch.Tensor":
    """Paste the final frame onto early frames: the target object appears early without contact."""
    B, C, T, H, W = z.shape
    out = z.clone()
    k = max(1, int(T * frac))
    out[:, :, :k] = z[:, :, -1:].expand(-1, -1, k, -1, -1)
    return out


def excision(z: "torch.Tensor", lo: float = 0.3, hi: float = 0.7) -> "torch.Tensor":
    """Excise the "contact->transport" middle segment, splice head and tail,
    resample back to the original length (missing necessary stage)."""
    B, C, T, H, W = z.shape
    a, b = int(T * lo), int(T * hi)
    kept = torch.cat([z[:, :, :a], z[:, :, b:]], dim=2)
    idx = torch.linspace(0, kept.shape[2] - 1, T, device=z.device).round().long()
    return kept[:, :, idx]


def shuffle_mid(z: "torch.Tensor", lo: float = 0.25, hi: float = 0.85) -> "torch.Tensor":
    """Shuffle the middle time block (causal order reversed)."""
    B, C, T, H, W = z.shape
    a, b = int(T * lo), int(T * hi)
    out = z.clone()
    perm = torch.randperm(b - a, device=z.device) + a
    out[:, :, a:b] = z[:, :, perm]
    return out


def freeze_jump(z: "torch.Tensor", frac: float = 0.55) -> "torch.Tensor":
    """First half frozen at the initial state, then a jump to near-final
    (premature final state + discontinuity)."""
    B, C, T, H, W = z.shape
    out = z.clone()
    k = int(T * frac)
    out[:, :, :k] = z[:, :, :1].expand(-1, -1, k, -1, -1)
    out[:, :, k:] = z[:, :, -1:].expand(-1, -1, T - k, -1, -1)
    return out


NEG_FUNCS = {"teleport": teleport, "excision": excision,
             "shuffle": shuffle_mid, "freeze_jump": freeze_jump}


def make_negatives(z: "torch.Tensor", kinds=("teleport", "excision", "freeze_jump")):
    """Return dict{kind: z^-}; 2-3 kinds built online per training batch."""
    return {k: NEG_FUNCS[k](z) for k in kinds}


def random_control(z: "torch.Tensor") -> "torch.Tensor":
    """Randomly permute all frames — ablation: if random negatives are equally
    useful, the signal is not causal."""
    B, C, T, H, W = z.shape
    return z[:, :, torch.randperm(T, device=z.device)]
