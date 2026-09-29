#!/usr/bin/env python3
"""EWM DUP 8-GPU dispatch + aggregation (called from a kjob; no bash wait)."""
import glob, json, os, subprocess, sys
from collections import defaultdict
HERE=os.path.dirname(os.path.abspath(__file__))
OUT=os.environ["EWM_OUT"]; os.makedirs(OUT,exist_ok=True)
procs=[]
for g in range(8):
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),PYTHONUNBUFFERED="1")
    procs.append(subprocess.Popen([sys.executable,f"{HERE}/ewm_dup.py",str(g),"8",f"{OUT}/part{g}.json"],env=env))
rc=0
for p in procs: rc|=p.wait()
recs=[]
for f in glob.glob(f"{OUT}/part*.json"): recs+=json.load(open(f))
by=defaultdict(list)
for r in recs: by[r["model"]].append(r)
print("=== EWMBench DUP (manual mover, inv0 condition image, per video) ===")
summ={}
for m,rs in sorted(by.items()):
    ok=[r for r in rs if not r["undetectable"]]
    app=[r for r in ok if r["applicable"]]
    da=sum(r["dup"] for r in ok)/max(1,len(ok)); dp=sum(r["dup"] for r in app)/max(1,len(app))
    summ[m]=dict(elig=len(ok),n=len(rs),dup_all=da,dup_app=dp,n_app=len(app))
    print(f"  {m:34s} elig={len(ok)}/{len(rs)} DUP(all)={da*100:.1f}% DUP(applicable)={dp*100:.1f}%")
json.dump(dict(summary=summ,records=recs),open(f"{OUT}/summary.json","w"))
sys.exit(rc)
