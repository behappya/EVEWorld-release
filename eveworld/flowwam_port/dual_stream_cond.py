#!/usr/bin/env python3
"""Optical-flow-conditioned dual-stream forward (for action/HDF5-driven inference).

Code-generated at import time from the official model_fn source: flow-stream per-token
timesteps are all 0 (flow latent pinned to a clean encoding externally = teacher forcing).
Exec'd in the ds module namespace so TIAInjection's monkeypatch still takes effect.
"""
from __future__ import annotations

import inspect

import diffsynth.pipelines.wan_video_dual_stream as ds

_FLOW_TPT_ORIG = """            flow_tpt = torch.cat([
                torch.zeros(1, flow_spatial, dtype=latents.dtype,
                            device=latents.device),
                torch.ones(flow_temporal - 1, flow_spatial, dtype=latents.dtype,
                           device=latents.device) * ts_b,
            ]).flatten()"""

_FLOW_TPT_COND = """            flow_tpt = torch.zeros(
                flow_temporal * flow_spatial, dtype=latents.dtype,
                device=latents.device)  # clean condition for all flow frames: t=0"""

_src = inspect.getsource(ds.model_fn_wan_video_dual_stream)
assert _FLOW_TPT_ORIG in _src, "upstream model_fn source changed; re-check the flow_tpt section"
_src = _src.replace(_FLOW_TPT_ORIG, _FLOW_TPT_COND)
_src = _src.replace("def model_fn_wan_video_dual_stream(",
                    "def model_fn_dual_stream_flowcond(")
exec(compile(_src, "<dual_stream_cond codegen>", "exec"), ds.__dict__)

model_fn_dual_stream_flowcond = ds.model_fn_dual_stream_flowcond
