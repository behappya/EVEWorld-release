#!/usr/bin/env python3
"""Gripper detection dispatch: shard videos across 8 GPUs (kjobctl static-scan compatible)."""
import os
import subprocess
import sys

ngpu, out_dir, tp = int(sys.argv[1]), sys.argv[2], sys.argv[3]
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
procs = []
for g in range(ngpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED='1')
    procs.append(subprocess.Popen(
        [tp, os.path.join(here, 'gripper_detect.py'),
         '--shard-index', str(g), '--num-shards', str(ngpu), '--out-dir', out_dir], env=env))
rc = 0
for p in procs:
    rc |= p.wait()
sys.exit(rc)
