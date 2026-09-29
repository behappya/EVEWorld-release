#!/usr/bin/env python3
"""Minimal distributed smoke test: one 8-rank NCCL all_reduce, with NCCL_DEBUG=INFO evidence."""
import os

import torch
import torch.distributed as dist

rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
print(f"rank{rank}: torch={torch.__version__} nccl={torch.cuda.nccl.version()}", flush=True)
dist.init_process_group(backend="nccl")
x = torch.ones(1024, 1024, device=f"cuda:{rank}")
dist.all_reduce(x)
print(f"rank{rank}: all_reduce OK sum={float(x[0, 0])}", flush=True)
dist.barrier()
if rank == 0:
    print("DIST_SMOKE_OK", flush=True)
dist.destroy_process_group()
