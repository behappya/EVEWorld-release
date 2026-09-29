#!/usr/bin/env python3
"""TEA layer-1 main metric: motion-dynamics laziness signatures (TELE/JUMP/STILL/ROUGH -> LAZY).

CPU-only (opencv-python + numpy); measured on real-vs-corruption: TELE/JUMP catch teleport, ROUGH catches freeze_jump.
"""
import argparse, glob, json, os, sys
import numpy as np

try:
    import cv2
except Exception:
    print("[tea] opencv-python required: pip install opencv-python-headless", file=sys.stderr)
    raise


def read_frames(path, max_frames=48, resize=192, crop=None):
    """Read a video -> grayscale frame list.
    crop=(lo,hi) selects a horizontal range (0.5,1.0 = right half of generated videos)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if crop is not None:
            w0 = f.shape[1]
            f = f[:, int(crop[0] * w0):int(crop[1] * w0)]
        h, w = f.shape[:2]
        s = resize / max(h, w)
        f = cv2.resize(f, (max(8, int(w * s)), max(8, int(h * s))))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    cap.release()
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames


def flow_seq(frames):
    """Sequence of per-adjacent-frame Farneback optical-flow magnitude maps."""
    out = []
    for t in range(len(frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(frames[t], frames[t + 1], None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        out.append(np.sqrt((flow ** 2).sum(-1)))
    return out


def score_frames(frames, tele_k=3.0, jump_k=4.0, rough_ref=0.45,
                 still_frac=0.1, weights=(0.4, 0.2, 0.2, 0.2)):
    """Compute laziness signatures for a frame sequence; returns dict, headline score LAZY."""
    T = len(frames)
    if T < 4:
        return None
    mags = flow_seq(frames)
    peaks = np.array([m.max() for m in mags], dtype=np.float32)
    energy = np.array([m.sum() for m in mags], dtype=np.float32)

    peak_med = float(np.median(peaks) + 1e-6)
    energy_med = float(np.median(energy) + 1e-6)

    TELE = float((peaks > tele_k * peak_med).mean())
    er = energy / energy_med
    JUMP = float((er > jump_k).mean())
    STILL = float((energy < still_frac * energy_med).mean())
    d = np.abs(np.diff(energy))
    ROUGH = float(d.mean() / (energy.mean() + 1e-6))

    # Normalize (ROUGH against the real baseline rough_ref; the rest are ratios in [0,1])
    rough_n = max(0.0, (ROUGH - rough_ref) / rough_ref)   # relative excess over real roughness
    w = weights
    LAZY = float(w[0] * TELE + w[1] * JUMP + w[2] * STILL + w[3] * min(1.0, rough_n))
    return {"n_frames": T, "LAZY": LAZY,
            "TELE": TELE, "JUMP": JUMP, "STILL": STILL, "ROUGH": ROUGH,
            "peak_med": peak_med, "energy_med": energy_med}


def score_video(path, crop=None, **kw):
    frames = read_frames(path, crop=crop)
    r = score_frames(frames, **kw)
    if r is None:
        return None
    r = {"video": os.path.basename(path), **r}
    return r


def corrupt(frames, kind):
    """Deterministic temporal corruption: same frames, only reordered/replaced,
    so per-frame quality stays identical."""
    T = len(frames)
    rng = np.random.RandomState(0)
    out = frames[:]
    if kind == "shuffle":
        mid = list(range(int(T * 0.25), int(T * 0.85)))
        perm = mid[:]; rng.shuffle(perm)
        for s, d in zip(mid, perm):
            out[d] = frames[s]
    elif kind == "reverse":
        out = frames[::-1]
    elif kind == "teleport":
        k = max(1, int(T * 0.4))
        for i in range(k):
            out[i] = frames[-1]
    elif kind == "freeze_jump":
        k = int(T * 0.55)
        for i in range(k):
            out[i] = frames[0]
        for i in range(k, T):
            out[i] = frames[-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--video-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--tag", default="model")
    s.add_argument("--crop", default=None, help="horizontal crop; right half uses 0.5,1.0")
    s.add_argument("--limit", type=int, default=0)

    v = sub.add_parser("validate")
    v.add_argument("--video-dir", required=True, help="real-video directory (anchor)")
    v.add_argument("--out", required=True)
    v.add_argument("--crop", default=None)
    v.add_argument("--limit", type=int, default=30)

    a = ap.parse_args()
    crop = tuple(float(x) for x in a.crop.split(",")) if a.crop else None
    vids = sorted(glob.glob(os.path.join(a.video_dir, "**", "*.mp4"), recursive=True))

    if a.cmd == "score":
        if a.limit:
            vids = vids[:a.limit]
        rows = []
        for i, vpath in enumerate(vids):
            r = score_video(vpath, crop=crop)
            if r:
                rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(vids)}")
        def m(key):
            return float(np.mean([r[key] for r in rows])) if rows else 0.0
        agg = {"tag": a.tag, "n": len(rows), "LAZY_mean": m("LAZY"),
               "TELE_mean": m("TELE"), "JUMP_mean": m("JUMP"),
               "STILL_mean": m("STILL"), "ROUGH_mean": m("ROUGH")}
        json.dump({"summary": agg, "per_video": rows}, open(a.out, "w"), indent=2)
        print(json.dumps(agg, indent=2, ensure_ascii=False)); print("wrote", a.out)
        return

    # validate
    if a.limit:
        vids = vids[:a.limit]
    kinds = ["real", "shuffle", "reverse", "teleport", "freeze_jump"]
    keys = ["LAZY", "TELE", "JUMP", "STILL", "ROUGH"]
    scores = {k: {kk: [] for kk in keys} for k in kinds}
    for i, vpath in enumerate(vids):
        base = read_frames(vpath, crop=crop)
        if len(base) < 4:
            continue
        for k in kinds:
            frames = base if k == "real" else corrupt(base[:], k)
            r = score_frames(frames)
            if r:
                for kk in keys:
                    scores[k][kk].append(r[kk])
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(vids)}")
    summary = {}
    real_lazy = np.array(scores["real"]["LAZY"])
    for k in kinds:
        summary[k] = {kk: float(np.mean(scores[k][kk])) for kk in keys}
        summary[k]["n"] = len(scores[k]["LAZY"])
        if k != "real":
            arr = np.array(scores[k]["LAZY"])
            if len(arr) == len(real_lazy) and len(real_lazy) > 1:
                diff = arr - real_lazy
                summary[k]["LAZY_gain_over_real"] = float(diff.mean())
                summary[k]["frac_higher_than_real"] = float((diff > 0).mean())
    json.dump({"summary": summary, "raw": scores}, open(a.out, "w"), indent=2)
    print(json.dumps(summary, indent=2, ensure_ascii=False)); print("wrote", a.out)
    print("\n[validity criterion] LAZY_mean of teleport/freeze_jump should be clearly > real;"
          " frac_higher_than_real closer to 1 is better; shuffle being smoother is expected"
          " (not a laziness proxy).")


if __name__ == "__main__":
    main()
