#!/usr/bin/env python3
"""Final combined arm v2 (arm 72 + isolation-verdict revision): pretrain base,
fully automatic closed loop, zero architectural change.
"""
from __future__ import annotations

import functools
import os

import numpy as np
import torch

from eveworld.pipeline.tia.trainer import T4GCorrTrainer, T4GCorrTransform  # noqa: F401 (transform registration)
from eveworld.pipeline.ich.ich_d_trainer import CIC_BLOCK, ICHDModule, NOVELTY_BLOCKS

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

T_LAT, H_LAT, W_LAT = 24, 30, 48

DEFAULT_WEIGHTS = f'{GAGI}/eve_v2_outputs/selfcase/cic_match/ich_d_frozen_lr.npz'


# pure functions (CPU-unit-testable)
def a2_sigma_schedule(k, s_hi=3.0, s_lo=0.3):
    """K large-sigma rounds: geometric coverage from high to low (same as eveworld.pipeline.probe.final_kprobe)."""
    return [float(s) for s in np.geomspace(s_hi, s_lo, k)]


def hit_upsample(hit):
    """hit (B,T,30,48) bool -> (B,1,T,60,96) float (latent-space loss mask)."""
    up = hit.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    return up.unsqueeze(1).float()


def pixel_prefill(pix, hit, feather=8):
    """Pixel-exact backfill (online version of the case-builder logic): each judge-marked cell's
    pixel span is replaced by the co-located block from that cell's "pre-onset" representative
    frame, with edge feathering.
    pix: (B,3,NF,Hp,Wp) fp32; hit: (B,T,30,48) bool; cell=16px.
    Returns (pix_fill, alpha (B,1,NF,Hp,Wp))."""
    import torch.nn.functional as TF
    B, C, NF, Hp, Wp = pix.shape
    T, Hc, Wc = hit.shape[1:]
    ch, cw = Hp // Hc, Wp // Wc
    rep = np.linspace(0, NF - 1, T).astype(int)
    mids = (rep[:-1] + rep[1:]) // 2 + 1
    seg_lo = np.concatenate([[0], mids])
    seg_hi = np.concatenate([mids, [NF]])
    out = pix.clone()
    hard = torch.zeros(B, 1, NF, Hp, Wp, device=pix.device, dtype=pix.dtype)
    last_clean = torch.zeros(B, Hc, Wc, dtype=torch.long, device=pix.device)
    for t in range(T):
        for b, gy, gx in torch.nonzero(hit[:, t]).tolist():
            tp = int(last_clean[b, gy, gx])               # pre-onset frame (fallback: frame 0 = ref)
            y0, x0 = gy * ch, gx * cw
            src = pix[b, :, rep[tp], y0:y0 + ch, x0:x0 + cw].unsqueeze(1)
            out[b, :, seg_lo[t]:seg_hi[t], y0:y0 + ch, x0:x0 + cw] = src
            hard[b, 0, seg_lo[t]:seg_hi[t], y0:y0 + ch, x0:x0 + cw] = 1.0
        cur = torch.full_like(last_clean, t)
        last_clean = torch.where(hit[:, t], last_clean, cur)
    # feathering (per-frame spatial box blur; clamp*1.2 keeps the core ~=1, edges linear)
    k = 2 * feather + 1
    kernel = torch.ones(1, 1, k, k, device=pix.device, dtype=pix.dtype) / (k * k)
    a2d = hard.reshape(B * NF, 1, Hp, Wp)
    a2d = TF.conv2d(TF.pad(a2d, (feather,) * 4, mode='replicate'), kernel)
    alpha = (a2d.reshape(B, 1, NF, Hp, Wp) * 1.2).clamp(0, 1)
    return alpha * out + (1 - alpha) * pix, alpha


def prefill_target(x0_det, hit):
    """Online erasure target: judge-marked cells are backfilled with the
    co-located x0 of the "pre-onset frame" (nearest unmarked frame per cell,
    fallback frame 0).
    x0_det: (B,z,T,60,96) detached latent;  hit: (B,T,30,48) bool.
    Returns (target with x0's shape, hit_up (B,1,T,60,96) float)."""
    B, T, H, W = hit.shape
    dev = hit.device
    tgt_idx = torch.zeros(B, T, H, W, dtype=torch.long, device=dev)
    last_clean = torch.zeros(B, H, W, dtype=torch.long, device=dev)   # frame 0 = ref, always clean
    for t in range(T):
        cur = torch.full_like(last_clean, t)
        tgt_idx[:, t] = torch.where(hit[:, t], last_clean, cur)
        last_clean = torch.where(hit[:, t], last_clean, cur)
    idx_up = tgt_idx.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    hit_up = hit.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    z = x0_det.shape[1]
    idx_g = idx_up.unsqueeze(1).expand(B, z, T, idx_up.shape[-2], idx_up.shape[-1])
    target = x0_det.gather(2, idx_g)
    return target, hit_up.unsqueeze(1).float()


# trainer
class T4GFinalTrainer(T4GCorrTrainer):
    """SFT 3/4 (L_diff+L_id) + A2 online snowball 1/4; frozen detector = train-side judge (v2)."""

    CORR_STATS_DIM = 11                     # stats length of t4g_corr._compute_corr

    # models
    def get_models(self, model_config):
        model = super().get_models(model_config)          # corr: L_id/annos/sentinels
        weights_path = os.environ.get('T4G_ICH_D_WEIGHTS',
                                      model_config.get('t4g_ich_d_weights', DEFAULT_WEIGHTS))
        self.a2_p = float(os.environ.get('T4G_A2_P', model_config.get('t4g_a2_p', 0.25)))
        self.a2_k = int(os.environ.get('T4G_A2_K', model_config.get('t4g_a2_k', 4)))
        self.a2_sigma_hi = float(os.environ.get('T4G_A2_SIGMA_HI',
                                                model_config.get('t4g_a2_sigma_hi', 3.0)))
        self.a2_sigma_lo = float(os.environ.get('T4G_A2_SIGMA_LO',
                                                model_config.get('t4g_a2_sigma_lo', 0.3)))
        self.lambda_a2 = float(os.environ.get('T4G_LAMBDA_A2',
                                              model_config.get('t4g_lambda_a2', 1.0)))
        self.a2_warmup = int(os.environ.get('T4G_A2_WARMUP',
                                            model_config.get('t4g_a2_warmup', 50)))
        self.a2_m_thr = float(os.environ.get('T4G_A2_M_THR',
                                             model_config.get('t4g_a2_m_thr', 0.7)))
        ich_win = int(os.environ.get('T4G_ICH_WIN', model_config.get('t4g_ich_win', 3)))

        # v2: frozen detector = pure train-side judge, a trainer attribute (not a
        # model submodule) - not in optimizer/EMA/checkpoint, zero model change,
        # so inference is the bare transformer.
        tf = model['transformer']
        judge = ICHDModule(channels=tf.config.model_channels, weights_path=weights_path,
                           win=ich_win)
        judge.to(self.device)                             # fp32 + GPU (features/scores only)
        judge.eval()
        judge.requires_grad_(False)
        self.a2_judge = judge
        self._judge_hook_done = False
        self._final_stat_buffer = []
        self._final_logged_step = -1
        self._a2_rng = np.random.default_rng(20260823 + int(os.environ.get('RANK', '0')))
        self.print('[t4g-final] v2 judge <- %s (frozen16d, train-side judge, not in model) | '
                   'a2: p=%.2f K=%d sigma=[%.1f->%.1f] lambda=%.2f warmup=%d '
                   'm_thr=%.2f | trainable=backbone only (zero model change)'
                   % (weights_path, self.a2_p, self.a2_k, self.a2_sigma_hi,
                      self.a2_sigma_lo, self.lambda_a2, self.a2_warmup, self.a2_m_thr))
        return model

    # hooks
    def _ensure_judge_hooks(self):
        """Judge observe hooks (read-only, no writeback; registered after corr hooks)."""
        if self._judge_hook_done:
            return
        model = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
        tf = model['transformer']
        judge = self.a2_judge
        for name in NOVELTY_BLOCKS + (CIC_BLOCK,):
            tf.blocks[name].register_forward_hook(
                lambda _m, _i, out, _n=name: judge.observe(_n, out))
        self._judge_hook_done = True
        self.print('[t4g-final] judge hooks -> observe %s (read-only; no suppress, '
                   'model forward unchanged)' % list(NOVELTY_BLOCKS + (CIC_BLOCK,)))

    # helpers
    def _lambda_a2_warmup(self):
        return self.lambda_a2 * min(1.0, self.cur_step / max(1, self.a2_warmup))

    def _sym_zero_corr_gather(self):
        """A2-branch symmetric collective: same shape/order as the corr._compute_corr gather."""
        stats = torch.zeros(self.CORR_STATS_DIM, device=self.device)
        gathered = self.accelerator.gather(stats.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._stat_buffer.append(gathered)            # corr sentinel buffer (all-zero row)

    # forward
    def forward_step(self, batch_dict):
        self._ensure_hook()                               # corr hooks (idempotent)
        self._ensure_judge_hooks()                        # judge read-only hooks
        judge = self.a2_judge
        judge.reset()

        is_a2 = bool(self._a2_rng.random() < self.a2_p)
        # final sentinel stats (cross-process additive):
        # [0]n_sft [1]n_a2 [2](empty) [3]Σm_a2(full map,R1) [4]Σhit_cells [5]n_a2_hit
        # [6]Σl_a2_raw [7]cnt_l_a2 [8](empty) [9](empty) [10]n_calls
        fstats = torch.zeros(11, device=self.device)
        if is_a2:
            losses = self._a2_forward(batch_dict, fstats)
            self._sym_zero_corr_gather()                  # symmetry: corr gather placeholder
        else:
            losses = super().forward_step(batch_dict)     # L_diff + L_id (gather included)
            fstats[0] += 1
        judge.reset()                                     # release feature refs (read-only judge)
        # collective symmetry iron rule (three smoke deadlocks pinned by py-spy):
        # giga_train.parse_losses gathers once per key of the losses dict — the two
        # branches have different key sets => a different number of collectives per
        # micro-step => mismatch with the backward allreduce => NCCL deadlock.
        # fix: a fixed key set (order = construction order, identical across ranks);
        # missing keys are filled with zero scalars.
        zero = torch.zeros((), device=self.device)
        canon = dict(l_diff=zero, l_id=zero, l_change=zero, l_a2=zero)
        canon.update(losses)
        losses = canon
        with torch.no_grad():
            fstats[10] += 1
        gathered = self.accelerator.gather(fstats.unsqueeze(0)).sum(0).detach().cpu()
        if self.is_main_process:
            self._final_stat_buffer.append(gathered)
        return losses

    # A2
    def _a2_rollout_step(self, x0, ref_latents, ref_masks, prompt_embeds, fps,
                         sigma, grad):
        """One round of self-feeding (input construction bit-identical to
        GigaWorld0Trainer.forward_step):
        x0 -> renoise(sigma, explicit) -> forward -> denoise -> frame 0 ref constant.
        no_grad rounds bypass the DeepSpeed engine (engine.forward call count defines the
        grad-accum boundary; extra calls drift ranks -> NCCL deadlock) — the gradient-
        carrying last round must go through the engine."""
        if grad:
            transformer = functools.partial(self.model, 'transformer')
        else:
            _m = self.accelerator.unwrap_model(self.model, keep_torch_compile=False)
            transformer = _m['transformer']               # AC wrapper kept, hooks intact
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            sig = torch.tensor([[sigma]], device=self.device, dtype=torch.float32)
            input_latents, timesteps = self.edm_loss.add_noise(x0.float(), sigma=sig)
            # dtype discipline: fp32 math -> unify to self.dtype before network input
            # (mixing breaks cat/linear)
            ref_m = ref_masks.float()
            augment_sigma = torch.tensor([0.0001], device=self.device,
                                         dtype=torch.float32)
            while len(augment_sigma.shape) < len(ref_latents.shape):
                augment_sigma = augment_sigma.unsqueeze(-1)
            input_latents = ref_m * ref_latents.float() + (1 - ref_m) * input_latents
            in_masks = ref_m.repeat(1, 1, 1, input_latents.shape[-2],
                                    input_latents.shape[-1])
            input_latents = torch.cat([input_latents, in_masks], dim=1)
            timesteps = timesteps.float().view(1, 1, 1, 1, 1).expand(
                x0.size(0), -1, x0.size(2), -1, -1)
            t_cond = augment_sigma / (augment_sigma + 1)
            timesteps = ref_m * t_cond + (1 - ref_m) * timesteps
            B = x0.shape[0]
            padding_mask = torch.zeros(
                (B, 1, x0.shape[-2] * 8, x0.shape[-1] * 8),
                dtype=self.dtype, device=self.device)
            pred = transformer(x=input_latents.to(self.dtype),
                               timesteps=timesteps.to(self.dtype),
                               crossattn_emb=prompt_embeds.to(self.dtype),
                               padding_mask=padding_mask, fps=fps)
            x0_new = self.edm_loss.denoise(pred.float())
            x0_new = ref_m.float() * ref_latents.float() + \
                (1 - ref_m.float()) * x0_new
        return x0_new

    def _a2_forward(self, batch_dict, fstats):
        """A2 sample: start from GT latents and roll K rounds (first K-1 no_grad);
        the judge marks emergent regions in the last-round x0 (M>0.7, last-round
        sigma=0.3 inside the judge calibration band); marked cells are backfilled
        with the pre-onset frame as target (detach) -> the repair loss is taught
        entirely to the backbone."""
        judge = self.a2_judge
        images = batch_dict['images']
        prompt_embeds = batch_dict['prompt_embeds']
        fps = batch_dict['fps'][0]
        latents_gt = self.forward_vae(images).float()
        ref_latents = self.forward_vae(batch_dict['ref_images'])
        ref_masks = batch_dict['ref_masks']

        sigmas = a2_sigma_schedule(self.a2_k, self.a2_sigma_hi, self.a2_sigma_lo)
        x0 = latents_gt
        for k, s in enumerate(sigmas):
            last = (k == len(sigmas) - 1)
            judge.reset()
            x0 = self._a2_rollout_step(x0, ref_latents, ref_masks, prompt_embeds,
                                       fps, s, grad=last)
            if not last:
                x0 = x0.detach()

        with torch.no_grad():
            m = judge.detect()                            # last-round features -> (B,T,30,48)
        hit = (m > self.a2_m_thr)
        n_hit = int(hit[:, 1:].sum())                     # frame 0 = ref, excluded
        hit = torch.cat([torch.zeros_like(hit[:, :1]), hit[:, 1:]], dim=1)
        lam = self._lambda_a2_warmup()
        if n_hit > 0:
            # pixel-exact target (align reconciliation FAIL fix directive): decode the
            # last-round x0 to pixels -> backfill the pre-onset frame in the lesion
            # region + feathering (same as the case builder) -> re-encode the whole clip.
            # = online version of the A1 dual-encoding mechanism; A2 samples only,
            # no_grad, the target is naturally detached.
            with torch.no_grad():
                raw = x0.detach().to(self.dtype) / self.latents_std + self.latents_mean
                pix = self.vae.decode(raw).sample.float().clamp(-1, 1)   # (B,3,NF,H,W)
                pix_fill, _ = pixel_prefill(pix, hit)
                z_tgt = self.forward_vae(
                    pix_fill.clamp(-1, 1).permute(0, 2, 1, 3, 4).to(self.dtype)).float()
            hit_up = hit_upsample(hit)
            err = (x0 - z_tgt) ** 2 * hit_up
            l_a2_raw = err.sum() / (hit_up.sum() * x0.shape[1] + 1e-8)
            with torch.no_grad():
                fstats[5] += 1
                fstats[6] += float(l_a2_raw.detach())
                fstats[7] += 1
        else:
            l_a2_raw = (x0 * 0).mean()                    # graph-keeping zero (no-hit sample)
        with torch.no_grad():
            fstats[1] += 1
            fstats[3] += float(m.mean())
            fstats[4] += n_hit
        return dict(l_a2=lam * l_a2_raw)

    # sentinel
    def print_step(self):
        super().print_step()                              # corr [t4g-sentinel]
        if not self.is_main_process or self.cur_step % self.log_interval != 0:
            return
        if self.cur_step == self._final_logged_step:
            return
        self._final_logged_step = self.cur_step
        agg = (torch.stack(self._final_stat_buffer).sum(0)
               if self._final_stat_buffer else torch.zeros(11))
        self._final_stat_buffer = []

        def _r(n, d):
            return (agg[n] / agg[d]).item() if agg[d] > 0 else float('nan')

        n_sft, n_a2, n_a2_hit = int(agg[0]), int(agg[1]), int(agg[5])
        msg = ('[a2-sentinel] step=%d sft=%d a2=%d a2_hit=%d | hit_cells/a2=%.1f '
               'l_a2=%.4e lambda_a2=%.3f K=%d | m_a2=%.4f (R1: sharp drop = alarm) '
               '| judge=frozen16d (v2, no eraser sub)'
               % (self.cur_step, n_sft, n_a2, n_a2_hit, _r(4, 1), _r(6, 7),
                  self._lambda_a2_warmup(), self.a2_k, _r(3, 1)))
        self.logger.info(msg)
        print(msg, flush=True)


# selftest (CPU): sigma schedule / prefill target / S5 static identity chain
def selftest():
    import functools as ft
    import operator as op
    torch.manual_seed(0)
    print('[final selftest] start')
    # sigma schedule
    s = a2_sigma_schedule(4)
    assert len(s) == 4 and s[0] == 3.0 and abs(s[-1] - 0.3) < 1e-9
    assert all(s[i] > s[i + 1] for i in range(3))

    # prefill target: hit cells backfill nearest unmarked frame (fallback frame 0);
    #    untouched cells target = self (zero loss)
    B, z, T, H, W = 1, 4, 6, 4, 6
    x0 = torch.arange(T).float().view(1, 1, T, 1, 1).expand(B, z, T, 2 * H, 2 * W).clone()
    hit = torch.zeros(B, T, H, W, dtype=torch.bool)
    hit[0, 3, 1, 2] = True                                # t=3 hit -> backfill t=2
    hit[0, 4, 1, 2] = True                                # t=4 consecutive hit -> still t=2
    hit[0, 2, 0, 0] = True                                # another cell t=2 -> backfill t=1
    tgt, hit_up = prefill_target(x0, hit)
    assert tgt.shape == x0.shape and hit_up.shape == (B, 1, T, 2 * H, 2 * W)
    assert float(tgt[0, 0, 3, 2, 4]) == 2.0 and float(tgt[0, 0, 4, 2, 4]) == 2.0
    assert float(tgt[0, 0, 2, 0, 0]) == 1.0
    assert float(tgt[0, 0, 5, 2, 4]) == 5.0               # not hit -> self
    assert float(hit_up.sum()) == 3 * 4                   # 3 hit cells x 2x2 upsample
    # all-hit column -> fallback frame 0
    hit2 = torch.zeros_like(hit); hit2[0, 1:, 2, 3] = True
    tgt2, _ = prefill_target(x0, hit2)
    assert all(float(tgt2[0, 0, t, 4, 6]) == 0.0 for t in range(1, T))
    print('[final selftest] prefill target OK')

    # 2b) pixel_prefill: marked pixel span replaced by co-located pre-onset block +
    #     feathering; outside stays bit-exact
    NF, Hp, Wp = 24, 64, 96
    Tq, Hc, Wc = 6, 4, 6
    pixv = torch.arange(NF).float().view(1, 1, NF, 1, 1).expand(1, 3, NF, Hp, Wp).clone()
    hitp = torch.zeros(1, Tq, Hc, Wc, dtype=torch.bool)
    hitp[0, 3, 1, 2] = True                               # t=3 -> t_pre=2
    pf, alpha = pixel_prefill(pixv, hitp, feather=4)
    rep = np.linspace(0, NF - 1, Tq).astype(int)
    mids = (rep[:-1] + rep[1:]) // 2 + 1
    slo = np.concatenate([[0], mids])[3]
    assert float(alpha[0, 0, slo, 24, 40]) > 0.99, 'backfilled core alpha should be ~1'
    assert float(alpha[0, 0, slo, 0, 0]) == 0.0, 'alpha outside should be 0'
    assert torch.equal(pf[0, :, :, 0:8, 0:8], pixv[0, :, :, 0:8, 0:8]), 'outside stays bit-exact'
    got = float(pf[0, 0, slo, 24, 40])
    want = float(rep[2])                                  # fill = representative frame of t_pre=2
    assert abs(got - want) < 0.6, (got, want)
    print(f'[final selftest] pixel_prefill OK (fill={got:.2f}≈rep[t_pre]={want}, '
          f'feathered edge, outside untouched)')

    # S5 v2: identity chain vs bare SFT with A2 off - judge is read-only observe
    #    (no suppress writeback, model forward unchanged) + canon key-set zero-fill
    #    does not change total_loss
    import json as _json
    import tempfile
    rngn = np.random.RandomState(0)
    tmp = tempfile.NamedTemporaryFile(suffix='.npz', delete=False)
    np.savez(tmp.name, w=rngn.randn(16), b=np.array([0.0]), mu=rngn.randn(16) * 0.1,
             sd=rngn.rand(16) + 0.5, meta=np.array(_json.dumps({}), dtype=object))
    judge = ICHDModule(channels=32, weights_path=tmp.name)
    judge.requires_grad_(False)
    for n, p in list(judge.named_parameters()) + list(judge.named_buffers()):
        assert p.dim() >= 1 and ft.reduce(op.mul, p.shape) == p.numel(), n
    feat = torch.randn(1, T_LAT, H_LAT, W_LAT, 32)
    for n in NOVELTY_BLOCKS + (CIC_BLOCK,):
        judge.observe(n, feat)                            # v2: read-only, no forward change
    with torch.no_grad():
        m = judge.detect()
    assert m.shape == (1, T_LAT, H_LAT, W_LAT) and not m.requires_grad
    # canon key set: zero-fill does not change total (numeric part of the S5 identity)
    l_diff = torch.randn(2).abs()
    losses = dict(l_diff=l_diff, l_id=torch.zeros(()), l_change=torch.zeros(()))
    zero = torch.zeros(())
    canon = dict(l_diff=zero, l_id=zero, l_change=zero, l_a2=zero)
    canon.update(losses)
    assert list(canon) == ['l_diff', 'l_id', 'l_change', 'l_a2'], list(canon)
    total_sft = sum(v.mean() for v in losses.values())
    total_canon = sum(v.mean() for v in canon.values())
    assert torch.equal(total_sft, total_canon), 'zero-fill canon keys must not change total_loss'
    os.unlink(tmp.name)
    print('[final selftest] S5 v2 (read-only judge + canon key-set identity) + R7 no-0dim OK')
    print('[final selftest] SELFTEST_OK')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    assert args.selftest, 'this file only supports --selftest; training enters via config runners'
    selftest()
