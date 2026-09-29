#!/usr/bin/env python3
"""EVE method: latent action model LAM (method §5-A, cluster-side torch): Genie-style inverse(z_t,z_{t+1})->discrete latent action code and forward_dyn(z_t,a)->z_{t+1}, in Wan VAE latent space (latent_channels=16)."""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class ConvEnc(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(256, cout)

    def forward(self, x):   # x:(N,cin,H,W)
        return self.fc(self.net(x).flatten(1))


class LatentActionModel(nn.Module):
    def __init__(self, latent_ch=16, action_dim=32, codebook=64):
        super().__init__()
        self.inv_enc = ConvEnc(latent_ch * 2, action_dim)     # action from (z_t,z_{t+1})
        self.codebook = nn.Parameter(torch.randn(codebook, action_dim))
        self.fwd = nn.Sequential(                              # (z_t,a) -> z_{t+1}
            nn.Conv2d(latent_ch, 128, 3, 1, 1), nn.SiLU())
        self.act_proj = nn.Linear(action_dim, 128)
        self.head = nn.Conv2d(128, latent_ch, 3, 1, 1)

    def _flatten_time(self, zt, ztp):
        B, C, T, H, W = zt.shape
        return (zt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W),
                ztp.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W), (B, T, H, W))

    def inverse(self, zt, ztp):
        """Returns the quantized latent action (B,T,action_dim)."""
        a, b, (B, T, H, W) = self._flatten_time(zt, ztp)
        raw = self.inv_enc(torch.cat([a, b], dim=1))          # (B*T, action_dim)
        # nearest-neighbor quantization to the codebook (straight-through)
        d = torch.cdist(raw, self.codebook)
        idx = d.argmin(1)
        q = self.codebook[idx]
        q = raw + (q - raw).detach()
        return q.view(B, T, -1)

    def forward_dyn(self, zt, a):
        """Given z_t (B,C,T,H,W) and action a (B,T,action_dim), predict z_{t+1}."""
        B, C, T, H, W = zt.shape
        x = zt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        h = self.fwd(x)
        av = self.act_proj(a.reshape(B * T, -1))[:, :, None, None]
        h = h + av
        out = self.head(h)
        return out.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    def transition_error(self, z):
        """Per-transition reconstruction error (B,T-1): lazy transitions score high.
        Used by eval layer 2 / the anti-shortcut regularizer."""
        zt, ztp = z[:, :, :-1], z[:, :, 1:]
        pred = self.forward_dyn(zt, self.inverse(zt, ztp))
        return ((pred - ztp) ** 2).mean(dim=(1, 3, 4))
