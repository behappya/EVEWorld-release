#!/usr/bin/env python3
"""EVE P0 - interactive CLI annotator (y/n/skip per dimension); one JSONL row per video, run once per annotator, then score with agreement.py."""
import argparse, json, os, glob


QS = [("premature", "did the task succeed before the required manipulation finished (early completion)?"),
      ("mbc", "does the object start moving before being contacted (moves before contact)?"),
      ("teleport", "does the object teleport / jump discontinuously?"),
      ("incomplete", "is a required stage missing (contact/grasp/transport/release)?"),
      ("not_executable", "is the whole process physically impossible?")]


def ask(q):
    while True:
        v = input(f"  {q} [y/n/s skip]: ").strip().lower()
        if v in ("y", "n", "s"):
            return {"y": 1, "n": 0, "s": None}[v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    done = set()
    if a.resume and os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["video"])
            except Exception:
                pass
    vids = sorted(glob.glob(os.path.join(a.video_dir, "**", "*.mp4"), recursive=True))
    fout = open(a.out, "a")
    print(f"{len(vids)} items, {len(done)} already labeled. Play each one in an external player.\n")
    for v in vids:
        key = os.path.basename(v)
        if key in done:
            continue
        print(f"\n=== {key} ===\n  path: {v}")
        rec = {"video": key, "path": v}
        for name, q in QS:
            rec[name] = ask(q)
        rec["lazy_any"] = int(any(rec[n] == 1 for n, _ in QS))
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        print("  recorded.")
    print("\nannotation done ->", a.out)


if __name__ == "__main__":
    main()
