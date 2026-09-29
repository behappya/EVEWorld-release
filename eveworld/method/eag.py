#!/usr/bin/env python3
"""EVE EAG executability-guided sampling (plan27 §4 I2): a differentiable soft top-k of the frozen LAD's transition_error guides denoising toward legal inter-frame transitions; sampling only, no retraining."""
from __future__ import annotations
import torch


class EAGGuidance:
    """Wrap a pretrained LAD as a sampling-time guidance energy + gradient."""

    def __init__(self, lam, topk: int = 3, tau: float = 0.5, weight: float = 0.0):
        self.lam = lam
        self.topk = topk
        self.tau = tau
        self.weight = weight
        for p in self.lam.parameters():
            p.requires_grad_(False)
        self.lam.eval()

    def energy(self, z0: torch.Tensor) -> torch.Tensor:
        """Transition executability energy E(z0). z0:(B,C,T,H,W). Returns a scalar (mean over B).
        top-k soft aggregation tracks the largest illegal jumps."""
        te = self.lam.transition_error(z0)            # (B, T-1)
        # soft top-k: per-sample weighted mean of the k largest transition errors (weights = softmax(te/tau))
        B = te.shape[0]
        k = min(self.topk, te.shape[1])
        topv, _ = te.topk(k, dim=1)                   # (B,k)
        w = torch.softmax(topv / self.tau, dim=1)     # (B,k)
        e_per = (w * topv).sum(dim=1)                 # (B,)  ~ soft-max
        return e_per.mean()

    @torch.enable_grad()
    def gradient(self, z0: torch.Tensor) -> torch.Tensor:
        """Returns dE/dz0, same shape as z0. Differentiated w.r.t. z0 only (LAD frozen, no second transformer forward)."""
        z0 = z0.detach().float().requires_grad_(True)
        e = self.energy(z0)
        grad, = torch.autograd.grad(e, z0, create_graph=False, retain_graph=False)
        return grad.detach()

    def sigma_weight(self, sigma: float, sigma_max: float = 80.0) -> float:
        """Schedule guidance strength by noise level: early high-noise x0 predictions are
        unreliable -> weak guidance, late low-noise -> strong. Monotone 1/(1+sigma).
        Measured: the per-step relative step must stay ~0.01-0.03 (w>=0.1 overshoots the
        minimum and diverges), so self.weight=0.02-0.05 keeps the decayed step safe."""
        return self.weight * (1.0 / (1.0 + float(sigma)))


def apply_eag_to_x0(z0_pred: torch.Tensor, guidance: EAGGuidance, sigma: float,
                    sigma_max: float = 80.0) -> torch.Tensor:
    """Apply EAG guidance to the x0 prediction (universal guidance on x0):
       z0' = z0 - w(sigma) * grad_z0 E(z0).
    Correcting x0 before letting the scheduler step pushes denoising toward legal transitions.
    """
    w = guidance.sigma_weight(sigma, sigma_max)
    if w <= 0:
        return z0_pred
    grad = guidance.gradient(z0_pred)                # dE/dz0 (gradient descent lowers the energy)
    # Step size: unit direction * (per-sample w * ||z0||). Relative to x0's own L2 norm,
    # w is "at most this fraction of x0 per step": bounded by construction, never diverges.
    B = grad.shape[0]
    g = grad.reshape(B, -1)
    gn = g / (g.norm(dim=1, keepdim=True) + 1e-8)     # unit direction (overall L2=1)
    gn = gn.reshape_as(grad)
    z0_norm = z0_pred.reshape(B, -1).norm(dim=1).view(B, 1, 1, 1, 1)
    return z0_pred - w * z0_norm * gn                # move w*||z0|| along the descent direction
