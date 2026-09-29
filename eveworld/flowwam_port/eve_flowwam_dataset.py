#!/usr/bin/env python3
"""EVE×FlowWAM shared training dataset for both arms (single-view WorldArena 640 tier).

One sample = a latent T_LAT_WIN window (4*(T-1)+1 pixel frames; frame0 is the condition
frame): clean window + (innovation arm) IGR paste-corrupted window + weight window + flow
codec video (RAFT full_scene) + prompt + TIA target cells. The last heldout_per_task
episodes per task are held out; RAFT is lazily initialized per dataloader worker.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
FLOWWAM_ROOT = os.environ.get("FLOWWAM_ROOT", os.path.expanduser("~/FlowWAM"))
for _p in (FLOWWAM_ROOT, os.path.join(FLOWWAM_ROOT, "training")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GAGI = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))

from igr_paste import apply_paste  # noqa: E402

N_LAT_EP, GH_W, GW_W = 31, 30, 40      # full-episode latent geometry (weightmap)
TOKEN_GH, TOKEN_GW = 15, 20            # DiT token grid


def lat_to_frame(t: int) -> int:
    return 0 if t == 0 else 4 * t - 1


class EVEFlowWAMWindowDataset:
    def __init__(
        self,
        manifest_path: str = f"{GAGI}/flowwam/igr/manifest_640.json",
        anno_dir: str = f"{GAGI}/flowwam/igr/anno_640",
        wmap_dir: str = f"{GAGI}/flowwam/igr/weightmap_cache_640",
        heldout_per_task: int = 5,
        split: str = "train",
        t_lat_win: int = 8,
        igr_paste: bool = False,
        paste_prob: float = 0.5,
        flow_mode: str = "full_scene",   # full_scene | robot_only (matches inference conditioning)
        flow_max_magnitude: float | None = None,
        samples_per_epoch: int | None = None,
        seed: int = 42,
        full_offset: bool = False,   # True: window start anywhere in episode (rolling inference)
    ) -> None:
        self.full_offset = full_offset
        rows = json.load(open(manifest_path))
        by_task: dict[str, list] = {}
        for r in rows:
            by_task.setdefault(r["task"], []).append(r)
        eps = []
        for task in sorted(by_task):
            lst = sorted(by_task[task], key=lambda r: int(r["episode"].replace("episode", "")))
            cut = len(lst) - heldout_per_task
            eps.extend(lst[:cut] if split == "train" else lst[cut:])
        self.episodes = eps
        self.anno_dir, self.wmap_dir = anno_dir, wmap_dir
        self.t_lat_win = int(t_lat_win)
        self.igr_paste = bool(igr_paste)
        self.paste_prob = float(paste_prob)
        assert flow_mode in ("full_scene", "robot_only"), flow_mode
        self.flow_mode = flow_mode
        self.flow_max_magnitude = flow_max_magnitude
        self.samples_per_epoch = samples_per_epoch or len(eps)
        self.seed = seed
        self._raft = None
        self._codec = None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _flow_tools(self):
        if self._raft is None:
            from raft_flow_extractor import RAFTFlowExtractor
            from reversible_flow_codec import FlowCodec
            self._raft = RAFTFlowExtractor(device="cuda")
            self._codec = FlowCodec()
        return self._raft, self._codec

    def _read_window(self, video: str, p0: int, n: int) -> np.ndarray:
        import cv2
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, p0)
        frames = []
        while len(frames) < n:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"empty window: {video}@{p0}")
        while len(frames) < n:
            frames.append(frames[-1].copy())
        return np.stack(frames)

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        row = self.episodes[index % len(self.episodes)]
        key = f"{row['task']}__{row['episode']}"
        anno = json.load(open(os.path.join(self.anno_dir, key + ".json")))
        wmap = np.load(os.path.join(self.wmap_dir, key + ".npy"))

        T = self.t_lat_win
        # some episodes are shorter than 121 frames: clamp the window start (pad tail frames)
        import cv2
        cap = cv2.VideoCapture(row["video"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if self.full_offset:
            # matches chunked rolling inference: start may sit anywhere in the episode
            n_lat_eff = max(T, (max(total, 2) - 1) // 4 + 1)
        else:
            n_lat_eff = max(T, min(N_LAT_EP, (max(total, 2) - 1) // 4 + 1))
        t0 = int(rng.integers(0, max(1, n_lat_eff - T + 1)))
        p0 = lat_to_frame(t0)
        n_px = 4 * (T - 1) + 1
        clean = self._read_window(row["video"], p0, n_px)
        w_win = wmap[t0:t0 + T].copy()
        if w_win.shape[0] < T:   # annotation covers only the first N_LAT_EP frames: pad uniformly
            pad = np.ones((T - w_win.shape[0],) + w_win.shape[1:], w_win.dtype)
            w_win = np.concatenate([w_win, pad], axis=0)

        corrupted, plan = clean, None
        if self.igr_paste and rng.random() < self.paste_prob:
            corrupted, w_win, plan = apply_paste(clean, w_win, anno, t0, rng)

        raft, codec = self._flow_tools()
        h, w = clean.shape[1:3]
        if self.flow_mode == "robot_only":
            # same conditioning as inference: robot_only render -> official process_camera_flow
            from flow_prefix_utils import process_camera_flow
            base = os.path.dirname(os.path.dirname(row["video"]))
            rv = os.path.join(base, "robot_only", "video", "head_camera",
                              row["episode"] + ".mp4")
            rframes = list(self._read_window(rv, p0, clean.shape[0]))
            flow_imgs, _ = process_camera_flow(
                rframes, (w, h), codec, flow_method="raft",
                raft_extractor=raft, max_magnitude=self.flow_max_magnitude)
            flow_imgs = flow_imgs[: clean.shape[0]]
        else:
            flow_imgs = [Image.new("RGB", (w, h), (255, 255, 255))]
            for f in range(1, clean.shape[0]):
                fl = raft(clean[f - 1], clean[f])
                rgb, _mag = codec.encode(fl, max_magnitude=self.flow_max_magnitude)
                flow_imgs.append(Image.fromarray(rgb))

        cells = []
        plf = anno["per_lat_frame"]
        for tl in range(T):
            c = plf[t0 + tl].get("target_cell") if t0 + tl < len(plf) else None
            cells.append(None if c is None else
                         (min(c[0] // 2, TOKEN_GH - 1), min(c[1] // 2, TOKEN_GW - 1)))

        return {
            "tiled_rgb_video": [[Image.fromarray(f) for f in clean]],
            "corrupted_rgb_video": [[Image.fromarray(f) for f in corrupted]],
            "flow_video": [flow_imgs],
            "video_prompt": row["instruction"],
            "action_prompt": row["instruction"],
            "weight_win": w_win.astype(np.float32),
            "tia_cells": cells,
            "meta": {"key": key, "t0": t0, "pasted": plan is not None},
        }
