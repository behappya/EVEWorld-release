#!/usr/bin/env python3
"""EVE P0 - event timeline extraction (CPU only, needs opencv-python + numpy): normalized event
times (0~1) and per-frame signals from a manipulation video, written as the JSON consumed by
process_metrics.py. Order-robust signals are preferred over precise coordinates.
"""
import argparse, json, os, glob, csv, sys
import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover
    print("[event_extract] needs opencv-python: pip install opencv-python-headless", file=sys.stderr)
    raise

N_BINS = 16  # number of bins on the event timeline


def read_frames(path, max_frames=256, resize=256, crop=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        # crop keeps a horizontal interval only, e.g. "0.5,1.0" takes the generated half of a side-by-side video
        if crop is not None:
            w0 = f.shape[1]
            f = f[:, int(crop[0] * w0):int(crop[1] * w0)]
        h, w = f.shape[:2]
        s = resize / max(h, w)
        f = cv2.resize(f, (int(w * s), int(h * s)))
        frames.append(f)
    cap.release()
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def foreground_motion(frames):
    """Per-frame foreground motion (difference vs first frame and vs previous frame, normalized). Returns (T,)."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    T = len(grays)
    motion = np.zeros(T, dtype=np.float32)
    for t in range(T):
        d0 = cv2.absdiff(grays[t], grays[0])
        dp = cv2.absdiff(grays[t], grays[max(0, t - 1)])
        d = np.maximum(d0, dp)
        motion[t] = float((d > 18).mean())
    return motion


def object_track_cheap(frames, prompt=None):
    """Cheap object localization: color word (if any) + motion-foreground component center. Returns (T,2) centers.
    This is the P0 baseline detector and is known to be imprecise; order-based metrics are robust to it.
    Can be replaced by CoTracker/SAM2 (see the optional track_cotracker implementation)."""
    T = len(frames)
    H, W = frames[0].shape[:2]
    centers = np.full((T, 2), np.nan, dtype=np.float32)
    prev = None
    for t in range(T):
        g = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(g, cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY))
        _, m = cv2.threshold(d, 18, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        # pick a large component that stays close to the previous center
        cand = np.argsort(areas)[::-1][:5] + 1
        best, bestcost = None, 1e9
        for c in cand:
            cx, cy = cent[c]
            cost = -areas[c - 1] / (H * W)
            if prev is not None:
                cost += 3.0 * np.hypot((cx - prev[0]) / W, (cy - prev[1]) / H)
            if cost < bestcost:
                bestcost, best = cost, (cx, cy)
        if best is not None:
            centers[t] = [best[0] / W, best[1] / H]
            prev = best
    return centers


def extract(video, prompt=None, crop=None):
    frames = read_frames(video, crop=crop)
    if len(frames) < 4:
        return None
    T = len(frames)
    motion = foreground_motion(frames)
    centers = object_track_cheap(frames, prompt)
    # object displacement speed (the signal that makes ordering robust)
    disp = np.zeros(T, dtype=np.float32)
    for t in range(1, T):
        if not np.isnan(centers[t]).any() and not np.isnan(centers[t - 1]).any():
            disp[t] = float(np.hypot(*(centers[t] - centers[t - 1])))
    # event heuristics on normalized time 0~1; thresholds can be calibrated on the labeled set later
    def first_ge(arr, thr):
        idx = np.where(arr >= thr)[0]
        return float(idx[0] / (T - 1)) if len(idx) else None
    mo_thr = max(0.02, float(np.nanpercentile(disp[disp > 0], 60)) if (disp > 0).any() else 0.02)
    events = {
        "t_action_start": first_ge(motion, max(0.02, motion.max() * 0.3)),
        "t_object_motion": first_ge(disp, mo_thr),
        "t_first_contact": None,   # approx: last rise of hand-object motion before the object displaces
        "t_success": None,         # approx: object arrives and settles (displacement falls back)
        "t_terminal_stable": None,
    }
    # contact approx: notable rise in motion before object_motion
    if events["t_object_motion"] is not None:
        om = int(events["t_object_motion"] * (T - 1))
        pre = motion[:max(1, om)]
        if len(pre):
            c = int(np.argmax(np.diff(pre))) if len(pre) > 1 else 0
            events["t_first_contact"] = float(c / (T - 1))
    # success/terminal approx: stable once displacement last exceeds the threshold
    moving = np.where(disp >= mo_thr)[0]
    if len(moving):
        last = int(moving[-1])
        events["t_success"] = float(last / (T - 1))
        events["t_terminal_stable"] = float(min(T - 1, last + 1) / (T - 1))
    # teleport signal: single-frame displacement jump
    jump = float(disp.max()) if T > 1 else 0.0
    return {
        "video": os.path.basename(video), "prompt": prompt, "n_frames": T, "n_bins": N_BINS,
        "events": events,
        "signals": {"motion": motion.tolist(), "obj_disp": disp.tolist(),
                     "centers": np.nan_to_num(centers, nan=-1).tolist()},
        "max_jump": jump,
        "detector": "cheap_cv_baseline_v1",
        "crop": crop,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--prompt")
    ap.add_argument("--video-dir"); ap.add_argument("--meta")
    ap.add_argument("--out"); ap.add_argument("--out-dir")
    ap.add_argument("--crop", default=None, help="horizontal crop interval, e.g. 0.5,1.0 keeps the right half")
    a = ap.parse_args()
    crop = tuple(float(x) for x in a.crop.split(',')) if a.crop else None
    if a.video:
        r = extract(a.video, a.prompt, crop=crop)
        out = a.out or (a.video + ".events.json")
        json.dump(r, open(out, "w"), indent=2)
        print("wrote", out); return
    assert a.video_dir and a.out_dir, "need --video-dir and --out-dir"
    os.makedirs(a.out_dir, exist_ok=True)
    prompts = {}
    if a.meta and os.path.exists(a.meta):
        for row in csv.DictReader(open(a.meta)):
            prompts[row.get("file_name", row.get("video", ""))] = row.get("text", row.get("prompt", ""))
    vids = sorted(glob.glob(os.path.join(a.video_dir, "**", "*.mp4"), recursive=True))
    print(f"found {len(vids)} videos")
    for i, v in enumerate(vids):
        key = os.path.basename(v)
        r = extract(v, prompts.get(key), crop=crop)
        if r is None:
            continue
        json.dump(r, open(os.path.join(a.out_dir, key + ".events.json"), "w"), indent=2)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(vids)}")
    print("done ->", a.out_dir)


if __name__ == "__main__":
    main()
