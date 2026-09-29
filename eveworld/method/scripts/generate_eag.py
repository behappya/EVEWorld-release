#!/usr/bin/env python3
"""EVE EAG sampling generation (plan 27 §4): runs the EAG pipeline on DreamGen inputs; --eag-weight 0 (plain sampling) vs >0 (EAG guidance) at the same seed for the paired laziness comparison; writes generated-only + side-by-side. Needs a GPU + train venv, run via kjob."""
import argparse, json, os, statistics, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # third_party/giga-world-0
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--lam", required=True, help="pretrained LAD checkpoint")
    ap.add_argument("--eag-weight", type=float, default=0.03, help="0=baseline (plain sampling)")
    ap.add_argument("--eag-topk", type=int, default=3)
    ap.add_argument("--eag-tau", type=float, default=0.5)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--num-inference-steps", type=int, default=30)
    ap.add_argument("--num-frames", type=int, default=93)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--seed", type=int, default=6666)
    ap.add_argument("--guidance-scale", type=float, default=7.0,
                    help="classifier-free guidance scale passed to the pipeline")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="catch-up mode: skip .mp4 already in generated_only, generate only missing")
    a = ap.parse_args()

    import torch, imageio
    from PIL import Image
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import functional as F
    from giga_datasets import image_utils
    from eveworld.method.pipeline_eag import EAGGigaWorld0Pipeline

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(a.save_dir, exist_ok=True)
    gen_dir = os.path.join(a.save_dir, "generated_only")
    sbs_dir = os.path.join(a.save_dir, "side_by_side")
    os.makedirs(gen_dir, exist_ok=True); os.makedirs(sbs_dir, exist_ok=True)

    print(f"[eag-gen] loading EAG pipeline (eag_weight={a.eag_weight})...", flush=True)
    pipe = EAGGigaWorld0Pipeline.from_pretrained(
        transformer_model_path=a.transformer,
        text_encoder_model_path=a.text_encoder,
        vae_model_path=a.vae,
        lora_model_path=a.lora,
        lora_fuse=bool(a.lora),
        physics_latent_model_path=None,     # EAG needs no physics token
    )
    pipe.to(dev)
    if a.eag_weight > 0:
        pipe.attach_eag(a.lam, weight=a.eag_weight, topk=a.eag_topk, tau=a.eag_tau)
    else:
        print("[eag-gen] eag_weight=0 -> baseline (plain sampling)", flush=True)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    neg = ("The video captures a series of frames showing ugly scenes, static with no motion, "
           "motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated, "
           "poorly lit, washed out colors, choppy, jerky movements, artifacting, unnatural transitions, "
           "fake elements, visual noise, flickering. Overall poor quality.")

    data = json.load(open(a.data_path))
    if a.limit:
        data = data[:a.limit]
    print(f"[eag-gen] {len(data)} samples -> {a.save_dir}", flush=True)

    timings = []
    t0 = time.time()
    for n in range(len(data)):
        d = data[n]
        prompt = d["prompt"]; image_path = d["image"]
        oid = f"{n}_" + "".join(c if c.isalnum() else "_" for c in prompt)[:80]
        gen_out = os.path.join(gen_dir, f"{oid}.mp4")
        sbs_out = os.path.join(sbs_dir, f"{oid}.mp4")
        if a.skip_existing and os.path.exists(gen_out) and os.path.exists(sbs_out):
            print(f"[eag-gen] {n+1}/{len(data)} {oid} SKIP(exists)", flush=True)
            continue
        if not os.path.exists(image_path):
            image_path = os.path.join(os.path.dirname(a.data_path), image_path)
        image = Image.open(image_path).convert("RGB")
        iw, ih = image.width, image.height
        dw, dh = image_utils.get_image_size((iw, ih), (a.width, a.height), mode="area", multiple=16)
        if float(dh) / ih < float(dw) / iw:
            nh = int(round(float(dw) / iw * ih)); nw = dw
        else:
            nh = dh; nw = int(round(float(dh) / ih * iw))
        x1 = (nw - dw) // 2; y1 = (nh - dh) // 2
        img = F.resize(image, (nh, nw), InterpolationMode.BILINEAR)
        img = F.crop(img, y1, x1, dh, dw)

        s = time.time()
        out = pipe(prompt=prompt, negative_prompt=neg, image=img,
                   guidance_scale=a.guidance_scale,
                   num_inference_steps=a.num_inference_steps, fps=a.fps,
                   num_frames=a.num_frames, height=dh, width=dw, seed=a.seed)[0]
        # generated-only
        imageio.mimsave(gen_out, list(out), fps=a.fps)
        # side-by-side (left: input image, right: generated)
        sbs = [image_utils.concat_images_grid([img, out[k]], cols=2, pad=2) for k in range(len(out))]
        imageio.mimsave(sbs_out, sbs, fps=a.fps)
        el = time.time() - s
        timings.append({"index": n, "output_id": oid, "elapsed_sec": round(el, 2)})
        print(f"[eag-gen] {n+1}/{len(data)} {oid} {el:.1f}s", flush=True)

    summary = {
        "data_path": a.data_path, "save_dir": a.save_dir,
        "generated_only_dir": gen_dir, "side_by_side_dir": sbs_dir,
        "eag_weight": a.eag_weight, "eag_topk": a.eag_topk, "eag_tau": a.eag_tau,
        "guidance_scale": a.guidance_scale,
        "lam": a.lam, "lora": a.lora, "num_frames": a.num_frames, "seed": a.seed,
        "count": len(timings), "wall_sec": round(time.time() - t0, 1),
        "mean_sec": round(statistics.mean([t["elapsed_sec"] for t in timings]), 2) if timings else None,
    }
    json.dump(summary, open(os.path.join(a.save_dir, "generation_summary.json"), "w"), indent=2)
    print(f"[eag-gen] DONE {len(timings)} videos -> {a.save_dir}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
