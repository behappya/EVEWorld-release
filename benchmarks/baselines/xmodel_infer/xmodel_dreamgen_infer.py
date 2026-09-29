#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batched cross-model I2V inference over a DreamGen manifest; writes one <request_id>.mp4 per item.
--model-family selects the diffusers pipeline: wan / wan_ti2v / cogvideox / cosmos.

Needs a GPU; on a GPU-less host only --dry-run works (plan + manifest check, no model)."""
import argparse
import json
import os
import sys
import time
import traceback


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-family", required=True,
                   choices=["wan", "wan_ti2v", "cogvideox", "cosmos"])
    p.add_argument("--model-path", required=True, help="local diffusers weights directory")
    p.add_argument("--data-path", required=True, help="DreamGen it2v.json (92 items)")
    p.add_argument("--save-dir", required=True, help="output directory (generated-only mp4)")
    p.add_argument("--num-frames", type=int, default=93)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=6666)
    p.add_argument("--data-limit", type=int, default=0,
                   help="only the first N items, 0=all (smoke)")
    p.add_argument("--negative-prompt", type=str, default="")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--dry-run", action="store_true",
                   help="only check inputs and print the plan, no model loading "
                        "(can run on a GPU-less host)")
    p.add_argument("--summary-path", type=str, default="")
    p.add_argument("--gpu-ids", type=str, default="0",
                   help="which GPUs to use for data parallelism, e.g. '0 1 2 3 4 5 6 7' "
                        "(92 items are split across them)")
    return p.parse_args()


def load_manifest(path, limit):
    with open(path, "r") as f:
        data = json.load(f)
    if limit and limit > 0:
        data = data[:limit]
    # Use request_id uniformly as the output file name (aligned with GW-0 generated-only)
    items = []
    for i, d in enumerate(data):
        rid = d.get("request_id") or d.get("id") or f"sample_{i}"
        img = d["image"]
        prompt = d.get("prompt", "")
        items.append({"request_id": rid, "image": img, "prompt": prompt})
    return items


def get_dtype(name):
    import torch
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def build_pipeline(family, model_path, dtype, device="cuda"):
    """Load the diffusers pipeline for the family and return (pipe, gen_fn).

    device is the GPU bound to this process, e.g. 'cuda:3' under data parallelism.
    """
    import torch
    from diffusers.utils import load_image

    if family in ("wan", "wan_ti2v"):
        # Both use WanImageToVideoPipeline; image_encoder is loaded from the repo when that subdir
        # exists (absent for TI2V-5B, which is officially supported). WanPipeline is pure T2V.
        from diffusers import WanImageToVideoPipeline as PipeCls
        pipe = PipeCls.from_pretrained(model_path, torch_dtype=dtype)
        pipe.to(device)

        def gen_fn(pipe, image, prompt, a, generator):
            kwargs = dict(
                image=image,
                prompt=prompt,
                negative_prompt=a.negative_prompt or None,
                height=a.height, width=a.width,
                num_frames=a.num_frames,
                num_inference_steps=a.num_inference_steps,
                guidance_scale=a.guidance_scale,
                generator=generator,
            )
            out = pipe(**kwargs)
            return out.frames[0]

        return pipe, gen_fn

    if family == "cogvideox":
        from diffusers import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe.to(device)
        pipe.vae.enable_tiling()

        def gen_fn(pipe, image, prompt, a, generator):
            out = pipe(
                image=image, prompt=prompt,
                num_frames=a.num_frames,
                num_inference_steps=a.num_inference_steps,
                guidance_scale=a.guidance_scale,
                generator=generator,
            )
            return out.frames[0]

        return pipe, gen_fn

    if family == "cosmos":
        from diffusers import Cosmos2VideoToWorldPipeline

        # Passthrough safety checker: the official cosmos_guardrail needs an online download,
        # and the inputs are fixed benchmark prompts.
        class _PassthroughSafetyChecker:
            def to(self, *args, **kwargs):
                return self

            def check_text_safety(self, *args, **kwargs):
                return True

            def check_video_safety(self, video, *args, **kwargs):
                return video

        pipe = Cosmos2VideoToWorldPipeline.from_pretrained(
            model_path, torch_dtype=dtype, safety_checker=_PassthroughSafetyChecker())
        pipe.to(device)

        def gen_fn(pipe, image, prompt, a, generator):
            out = pipe(
                image=image, prompt=prompt,
                height=a.height, width=a.width,
                num_frames=a.num_frames,
                num_inference_steps=a.num_inference_steps,
                guidance_scale=a.guidance_scale,
                generator=generator,
            )
            return out.frames[0]

        return pipe, gen_fn

    raise ValueError(f"unknown family: {family}")


def run_worker(rank, gpu_ids, all_items, args):
    """One GPU process: binds to a GPU, runs the all_items[rank::N] shard, writes videos + summary."""
    import torch
    from diffusers.utils import load_image, export_to_video

    n = len(gpu_ids)
    gid = gpu_ids[rank]
    torch.cuda.set_device(gid)
    dev = f"cuda:{gid}"
    my_items = all_items[rank::n]
    tag = f"[rank{rank}/{n} gpu{gid}]"
    print(f"{tag} got {len(my_items)}/{len(all_items)} items", flush=True)

    # Empty shard (e.g. 4 items on 8 GPUs, rank>=4): skip model loading, write an empty shard and exit.
    if len(my_items) == 0:
        with open(os.path.join(args.save_dir, f".shard_{rank}.json"), "w") as f:
            json.dump({"rank": rank, "gpu": gid, "n_ok": 0, "n_total": 0, "results": []}, f)
        print(f"{tag} no shard, skipping", flush=True)
        return

    dtype = get_dtype(args.dtype)
    t0 = time.time()
    print(f"{tag} loading {args.model_family} pipeline ...", flush=True)
    pipe, gen_fn = build_pipeline(args.model_family, args.model_path, dtype, device=dev)
    print(f"{tag} loaded in {time.time()-t0:.1f}s", flush=True)

    results = []
    n_ok = 0
    for j, it in enumerate(my_items):
        rid = it["request_id"]
        out_path = os.path.join(args.save_dir, f"{rid}.mp4")
        if os.path.exists(out_path):
            print(f"{tag} [skip] {j+1}/{len(my_items)} {rid}", flush=True)
            n_ok += 1
            results.append({"request_id": rid, "status": "exists", "path": out_path})
            continue
        try:
            image = load_image(it["image"]).resize((args.width, args.height))
            generator = torch.Generator(device=dev).manual_seed(args.seed)
            ts = time.time()
            frames = gen_fn(pipe, image, it["prompt"], args, generator)
            export_to_video(frames, out_path, fps=args.fps)
            dt = time.time() - ts
            n_ok += 1
            print(f"{tag} [ok] {j+1}/{len(my_items)} {rid} ({dt:.1f}s)", flush=True)
            results.append({"request_id": rid, "status": "ok", "path": out_path, "sec": dt})
        except Exception as e:
            print(f"{tag} [FAIL] {j+1}/{len(my_items)} {rid}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            results.append({"request_id": rid, "status": "fail", "error": str(e)})

    shard_path = os.path.join(args.save_dir, f".shard_{rank}.json")
    with open(shard_path, "w") as f:
        json.dump({"rank": rank, "gpu": gid, "n_ok": n_ok,
                   "n_total": len(my_items), "results": results}, f, ensure_ascii=False)
    print(f"{tag} done ok={n_ok}/{len(my_items)}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    items = load_manifest(args.data_path, args.data_limit)
    gpu_ids = [int(x) for x in str(args.gpu_ids).replace(",", " ").split()]

    print("=" * 60)
    print(f"[xmodel-infer] family={args.model_family}")
    print(f"  model_path = {args.model_path}")
    print(f"  data_path  = {args.data_path}  ({len(items)} items)")
    print(f"  save_dir   = {args.save_dir}")
    print(f"  size       = {args.width}x{args.height}  frames={args.num_frames} fps={args.fps}")
    print(f"  steps={args.num_inference_steps} cfg={args.guidance_scale} seed={args.seed} dtype={args.dtype}")
    print(f"  gpu_ids    = {gpu_ids}  (data parallel over {len(gpu_ids)} GPUs)")
    print("=" * 60)

    missing = [it for it in items if not os.path.exists(it["image"])]
    if missing:
        print(f"[WARN] {len(missing)} image paths do not exist, e.g.: {missing[0]['image']}")
    if not os.path.isdir(args.model_path):
        print(f"[WARN] model_path does not exist: {args.model_path}")

    if args.dry_run:
        print(f"[dry-run] OK: {len(items)} items to generate, {len(gpu_ids)} GPUs, first 3:")
        for it in items[:3]:
            print(f"    {it['request_id']}  <- {os.path.basename(it['image'])}")
        print("[dry-run] No model loaded, no inference run.")
        return 0

    import torch.multiprocessing as mp
    t0 = time.time()
    for r in range(len(gpu_ids)):
        p = os.path.join(args.save_dir, f".shard_{r}.json")
        if os.path.exists(p):
            os.remove(p)

    if len(gpu_ids) == 1:
        run_worker(0, gpu_ids, items, args)
    else:
        mp.start_processes(run_worker, nprocs=len(gpu_ids),
                           args=(gpu_ids, items, args), start_method="spawn")

    results, n_ok = [], 0
    for r in range(len(gpu_ids)):
        p = os.path.join(args.save_dir, f".shard_{r}.json")
        if os.path.exists(p):
            sd = json.load(open(p))
            results.extend(sd["results"])
            n_ok += sd["n_ok"]
    summary = {
        "model_family": args.model_family, "model_path": args.model_path,
        "total": len(items), "ok": n_ok, "gpu_ids": gpu_ids,
        "num_frames": args.num_frames, "fps": args.fps,
        "height": args.height, "width": args.width,
        "steps": args.num_inference_steps, "seed": args.seed,
        "wall_sec": time.time() - t0, "results": results,
    }
    sp = args.summary_path or os.path.join(args.save_dir, "generation_summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[done] {n_ok}/{len(items)} generated ok (took {time.time()-t0:.0f}s), summary: {sp}")
    return 0 if n_ok == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
