#!/usr/bin/env python3
"""EVE · Track4Gen 式对应监督 —— 两道实证 loss 纯函数实现 (43 号 §6 精确公式)。

L_id: block22 前后帧接力 (治瞬移/形变; E4 EPE0.55); L_change: block25 前后帧自相似 (治复制/重生; E7 gap+0.45)。
输入 fn_* 已单位化 (T,H,W,D), anno 取 t4g_anno/<vid>.json; 输出 {losses, counts, gate, diag}; 前后帧漏检帧跳过。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# 43 号 §6 起步超参 (不擅自改; trainer/env 可覆盖)
TAU = 0.07          # L_id softmax 温度
TOL = 0.2           # L_change 静止容差 (E7: 静止地板 self-sim 0.80 → 正常变化 0.20)
WIN_R_ID = 3        # L_id 局部窗口半径 (7×7; EPE0.55 → 物体 <1 格/帧, r=3 足够含目标)
OBJ_EXCL_R = 1      # L_change 物体足迹排除半径 (3×3, 覆盖物体格及上一帧离开位)


# 小工具 (纯坐标, 无梯度)
def _valid_cells(cells, H, W):
    out = []
    for c in cells or []:
        gy, gx = int(c[0]), int(c[1])
        if 0 <= gy < H and 0 <= gx < W:
            out.append((gy, gx))
    return out


def _neighborhood(cell, H, W, r):
    gy, gx = int(cell[0]), int(cell[1])
    out = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ny, nx = gy + dy, gx + dx
            if 0 <= ny < H and 0 <= nx < W:
                out.append((ny, nx))
    return out


def prepare(feat, eps=1e-8):
    """L2 归一化特征 (沿 D), 返回 (fn, meta)。fn 与 feat 连图 (归一化可导)。"""
    T, H, W, D = feat.shape
    fn = feat / (feat.norm(dim=-1, keepdim=True) + eps)
    return fn, dict(T=T, H=H, W=W, D=D)


# ① 身份 loss L_id (block22, 前后帧接力, 治瞬移/形变)
def loss_id(fn, anno, meta, tau=TAU, win_r=WIN_R_ID):
    """前后帧接力 softmax-CE (τ 温度): 上一帧物体格特征 q, 在上一帧位置的 win_r 邻域窗口内, 应对当前帧目标格得分最高。
    参照 q=fn[t-1, target_cell_{t-1}] (非第0帧全局); 候选=邻域−distractor (目标格必留); 漏检/移出窗口→跳过。"""
    H, W = meta['H'], meta['W']
    device = fn.device
    per = anno['per_lat_frame']
    Tn = min(meta['T'], len(per))
    distractor = set(_valid_cells(anno.get('distractor_cells', []), H, W))
    fn_flat = fn.reshape(fn.shape[0], H * W, -1)

    terms = []
    n_skip = 0
    for t in range(1, Tn):
        fp, fc = per[t - 1], per[t]
        if not (fp.get('detected', False) and fc.get('detected', False)):
            continue
        pc, cc = fp.get('target_cell'), fc.get('target_cell')
        if pc is None or cc is None:
            continue
        pgy, pgx, cgy, cgx = int(pc[0]), int(pc[1]), int(cc[0]), int(cc[1])
        if not (0 <= pgy < H and 0 <= pgx < W and 0 <= cgy < H and 0 <= cgx < W):
            continue
        # 上一帧位置的局部窗口, 排除 distractor (目标格例外必留)
        win = _neighborhood((pgy, pgx), H, W, r=win_r)
        cand = [c for c in win if (c not in distractor or c == (cgy, cgx))]
        if (cgy, cgx) not in cand:
            n_skip += 1                                   # 目标移出窗口 (瞬移超窗) -> 无法监督
            continue
        q = fn[t - 1, pgy, pgx]                            # (D,) 上一帧物体指纹 (已单位化)
        idx = torch.tensor([c[0] * W + c[1] for c in cand], device=device, dtype=torch.long)
        logits = (fn_flat[t][idx] @ q) / tau              # (n,) 当前帧窗口内各格余弦/τ
        logp = F.log_softmax(logits, dim=0)
        terms.append(-logp[cand.index((cgy, cgx))])
    diag = dict(id_win_skipped=n_skip)
    if not terms:
        return fn.new_zeros(()), 0, diag
    return torch.stack(terms).mean(), len(terms), diag


# ② 变化 loss L_change (block25, 前后帧自相似, 治复制/重生; 主力)
def loss_change(fn, anno, meta, tol=TOL, obj_r=OBJ_EXCL_R):
    """该静组前后帧自相似违背: mean relu((1−cos(feat[t,c], feat[t-1,c])) − tol)。
    static_t = 全格 − 物体足迹 (当前+上一帧 target_cell 各 obj_r 邻域) − (t>=t_arrival 后剔除的 B 格);
    仅 t<t_arrival 时 B 格留 static (框重叠闸); 漏检帧跳过, 从 t=1 起。返回 (loss, n_frames, gate)。"""
    H, W = meta['H'], meta['W']
    device = fn.device
    per = anno['per_lat_frame']
    Tn = min(meta['T'], len(per))
    t_arr = anno.get('t_arrival', None)
    t_arr = int(t_arr) if t_arr is not None else None
    fn_flat = fn.reshape(fn.shape[0], H * W, -1)

    terms = []
    b_active_frames = 0            # B 受约束帧 (t<t_arrival)
    b_released_frames = 0          # B 放开帧 (t>=t_arrival / 无 t_arrival)
    static_total = 0
    for t in range(1, Tn):
        fp, fc = per[t - 1], per[t]
        if not (fp.get('detected', False) and fc.get('detected', False)):
            continue
        pc, cc = fp.get('target_cell'), fc.get('target_cell')
        if pc is None or cc is None:
            continue
        static = torch.ones(H * W, dtype=torch.bool, device=device)
        # 物体足迹: 当前 + 上一帧 target_cell 各 obj_r 邻域 -> 剔除 (合法运动, 含离开位)
        for cell in ((int(pc[0]), int(pc[1])), (int(cc[0]), int(cc[1]))):
            for (yy, xx) in _neighborhood(cell, H, W, r=obj_r):
                static[yy * W + xx] = False
        # 框重叠闸: B 格
        b = _valid_cells(fc.get('b_cells', []), H, W)
        b_constrained = (t_arr is not None) and (t < t_arr)
        if b:
            if b_constrained:
                b_active_frames += 1                       # 留在 static (到达前, 该静)
            else:
                b_released_frames += 1                      # 剔除 (到达后放开, 不罚合法放置)
                for (yy, xx) in b:
                    static[yy * W + xx] = False
        idx = static.nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            continue
        cos = (fn_flat[t][idx] * fn_flat[t - 1][idx]).sum(-1)   # (n,) 前后帧自相似
        terms.append(F.relu((1.0 - cos) - tol).mean())
        static_total += int(idx.numel())
    gate = dict(b_active_frames=b_active_frames, b_released_frames=b_released_frames,
                static_cells_mean=(static_total / len(terms)) if terms else 0.0)
    if not terms:
        return fn.new_zeros(()), 0, gate
    return torch.stack(terms).mean(), len(terms), gate


# 编排: 一次算两道 (trainer 调用); L_id 吃 block22 特征, L_change 吃 block25 特征
def compute_corr_losses(feat_id, feat_change, anno, tau=TAU, tol=TOL,
                        win_r=WIN_R_ID, obj_r=OBJ_EXCL_R):
    """feat_id/feat_change (T,H,W,D) + anno -> {losses, counts, gate, diag}; 单层测试可传同一特征给两参 (smoke 用)。"""
    fn_id, meta_id = prepare(feat_id)
    fn_ch, meta_ch = prepare(feat_change)
    L_id, n_id, diag = loss_id(fn_id, anno, meta_id, tau=tau, win_r=win_r)
    L_change, n_change, gate = loss_change(fn_ch, anno, meta_ch, tol=tol, obj_r=obj_r)
    return dict(
        losses=dict(L_id=L_id, L_change=L_change),
        counts=dict(n_id=n_id, n_change=n_change),
        gate=gate,
        diag=dict(id_win_skipped=diag['id_win_skipped'],
                  static_cells_mean=gate['static_cells_mean']),
    )
