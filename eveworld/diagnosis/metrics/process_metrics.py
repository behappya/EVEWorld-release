#!/usr/bin/env python3
"""EVE P0 - process faithfulness metrics (CPU only) over a directory of *.events.json from
event_extract.py: PCR (success before the process finishes), MBC (motion before contact),
TELE (displacement jump), COMP (all required events), CBE (causal order) - the paper's main C1 metrics.
"""
import argparse, json, glob, os
import numpy as np

TELEPORT_THR = 0.15   # normalized single-frame displacement above this counts as a jump (calibrate on the labeled set)


def per_video(ev):
    e = ev["events"]
    disp = np.array(ev["signals"]["obj_disp"], dtype=np.float32)
    res = {"video": ev["video"]}
    # Teleport: any single-frame displacement over threshold
    res["teleport"] = int(ev.get("max_jump", 0.0) >= TELEPORT_THR)
    # Motion-before-contact: object moves before contact
    tc, tm = e.get("t_first_contact"), e.get("t_object_motion")
    res["mbc"] = int(tc is not None and tm is not None and tm < tc - 1e-6)
    # Premature completion: success before the process needed for a stable final state; approximated by
    #   whether success happens too early. The lazy pattern is the object "arriving" without a full transport.
    ts = e.get("t_success")
    res["premature"] = int(ts is not None and ts < 0.5 and disp[: max(1, int(ts * (len(disp) - 1)))].sum() < 0.3 * (disp.sum() + 1e-6))
    # Completeness: are all required events present (action_start/object_motion/success detected)
    need = ["t_action_start", "t_object_motion", "t_success"]
    res["complete"] = int(all(e.get(k) is not None for k in need))
    # Cause-before-Effect: contact before motion, motion before success
    order_ok = 1
    seq = [("t_first_contact", "t_object_motion"), ("t_object_motion", "t_success")]
    for a, b in seq:
        if e.get(a) is not None and e.get(b) is not None and e[a] > e[b] + 1e-6:
            order_ok = 0
    res["cbe"] = order_ok
    # overall laziness flag (any severe signal)
    res["lazy"] = int(res["teleport"] or res["mbc"] or res["premature"] or (not res["complete"]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="model")
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.events_dir, "*.events.json")))
    rows = []
    for f in files:
        try:
            rows.append(per_video(json.load(open(f))))
        except Exception as ex:
            print("skip", f, ex)
    n = max(1, len(rows))
    agg = {
        "tag": a.tag, "n_videos": len(rows),
        "PCR_premature": sum(r["premature"] for r in rows) / n,
        "MBC_motion_before_contact": sum(r["mbc"] for r in rows) / n,
        "TELE_teleport": sum(r["teleport"] for r in rows) / n,
        "COMP_completeness": sum(r["complete"] for r in rows) / n,
        "CBE_cause_before_effect": sum(r["cbe"] for r in rows) / n,
        "LAZINESS_RATE": sum(r["lazy"] for r in rows) / n,
    }
    json.dump({"summary": agg, "per_video": rows}, open(a.out, "w"), indent=2)
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
