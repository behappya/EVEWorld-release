#!/usr/bin/env python3
"""Extract the AgiBotWorld tar shards into the short sub-action clip trainset
agibot_ewm_train/<task>_<episode>_<segidx>.{mp4,txt}.

One clip per task_info action_config segment (93 sampled frames at 640x480 / 16fps); the 21
EWMBench test episodes are dropped as a leak guard and audited to disk.
"""
import glob
import json
import os
import tarfile
from multiprocessing import Pool

import cv2
import numpy as np

GAGI = os.environ.get("GAGI_ROOT", os.path.expanduser("~/gagi"))
RAW = f"{GAGI}/agibot_ewm_raw"
OUT = f"{GAGI}/agibot_ewm_train"
TMP = f"{GAGI}/agibot_ewm_raw/_extract_tmp"
TEST_EPS = {  # 21 EWMBench gt_dataset test episodes (leak guard: drop them all)
    "649524", "649559", "650191", "651464", "664600", "681186",
    "766602", "773025", "773496", "743247", "743964", "744776",
    "798615", "798749", "807480", "787136", "789120", "791059",
    "808158", "824748", "834014",
}
N_FRAMES, FPS, W, H = 93, 16, 640, 480
MIN_SEG_FRAMES = 40          # drop sub-segments too short (<40 frames@30fps ≈1.3s)
MAX_SEG_PER_EP = 6


def resample(n, k):
    return [min(n - 1, round(i * (n - 1) / (k - 1))) for i in range(k)]


def one_episode(args):
    task, ep, mp4_path, segs = args
    made = 0
    try:
        import av  # PyAV(libdav1d): AgiBotWorld videos are AV1, cv2 cannot decode
        container = av.open(mp4_path)
        frames = [fr.to_ndarray(format="bgr24") for fr in container.decode(video=0)]
        container.close()
        total = len(frames)
        for si, seg in enumerate(segs[:MAX_SEG_PER_EP]):
            s, e, text = seg["start_frame"], seg["end_frame"], seg["action_text"].strip()
            e = min(e, len(frames))
            if e - s < MIN_SEG_FRAMES or not text:
                continue
            name = f"{task}_{ep}_{si}"
            vp = f"{OUT}/{name}.mp4"
            if os.path.exists(vp):
                made += 1
                continue
            sub = frames[s:e]
            vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
            for i in resample(len(sub), N_FRAMES):
                vw.write(cv2.resize(sub[i], (W, H), interpolation=cv2.INTER_AREA))
            vw.release()
            with open(f"{OUT}/{name}.txt", "w") as f:
                f.write(text)
            made += 1
        return f"{task}/{ep}", made, total
    except Exception as e:  # noqa: BLE001
        return f"{task}/{ep}", f"FAIL {type(e).__name__}: {e}", 0


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    # instruction index: task -> episode -> [segments]
    seg_index = {}
    for jf in glob.glob(f"{RAW}/task_info/task_*.json"):
        task = os.path.basename(jf).replace("task_", "").replace(".json", "")
        for item in json.load(open(jf)):
            ep = str(item.get("episode_id"))
            segs = (item.get("label_info") or {}).get("action_config") or []
            seg_index.setdefault(task, {})[ep] = segs

    audit = {"excluded_test_eps": [], "extracted": [], "missing_seg": []}
    jobs = []
    for tar_path in sorted(glob.glob(f"{RAW}/observations/*/*.tar")):
        task = tar_path.split("/")[-2]
        with tarfile.open(tar_path) as tf:
            members = [m for m in tf.getmembers()
                       if m.name.endswith("videos/head_color.mp4")]
            for m in members:
                ep = m.name.split("/")[0]
                if ep in TEST_EPS:
                    audit["excluded_test_eps"].append(f"{task}/{ep}")
                    continue
                segs = seg_index.get(task, {}).get(ep)
                if not segs:
                    audit["missing_seg"].append(f"{task}/{ep}")
                    continue
                dst = f"{TMP}/{task}_{ep}_head.mp4"
                if not os.path.exists(dst):
                    with tf.extractfile(m) as src, open(dst, "wb") as out:
                        out.write(src.read())
                jobs.append((task, ep, dst, segs))
    print(f"episodes to extract: {len(jobs)}, test-set excluded: {len(audit['excluded_test_eps'])}")

    n_clip = 0
    with Pool(8) as p:
        for key, made, total in p.imap_unordered(one_episode, jobs):
            if isinstance(made, int):
                n_clip += made
                audit["extracted"].append({"ep": key, "clips": made, "src_frames": total})
            else:
                print(f"  {key}: {made}", flush=True)
    json.dump(audit, open(f"{OUT}/_extract_audit.json", "w"), indent=1)
    print(f"total clips: {n_clip} -> {OUT} (audit _extract_audit.json)")


if __name__ == "__main__":
    main()
