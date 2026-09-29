#!/usr/bin/env python3
"""GroundingDINO object localisation: parse object names from the prompt -> box them in the first
frame -> feature-grid coords. Gives t4g probes/viz a correct tracking seed, not the flow argmax."""
import os
import re

import numpy as np

GDINO_PATH = f"{os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))}/ext_data/grounding_dino_base"


def parse_objects(prompt):
    """Parse X (moved object) / A,B (source/target containers, static) from a template prompt:
    "Use the {L/R} hand to pick up {X} from {A} to {B}". Returns a dict."""
    m = re.search(r'pick up (?:the )?(.+?) from (?:the )?(.+?) to (?:the )?(.+?)\.?\s*$', prompt, re.I)
    if m:
        return {'mover': m.group(1).strip(), 'src': m.group(2).strip(), 'tgt': m.group(3).strip()}
    # fallback: first noun phrase after "pick up"
    m2 = re.search(r'pick up (?:the )?(.+?)(?: from| to|\.|\s*$)', prompt, re.I)
    return {'mover': m2.group(1).strip() if m2 else prompt, 'src': None, 'tgt': None}


class GDinoLocator:
    def __init__(self, device='cuda'):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(GDINO_PATH)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_PATH).to(device).eval()
        self.device = device
        self.torch = torch

    def locate(self, frame_rgb, text, box_thr=0.20, text_thr=0.15):
        """frame_rgb: HxWx3 uint8; text: object name. Returns (cx, cy, box) in pixels, or None."""
        from PIL import Image
        torch = self.torch
        img = Image.fromarray(frame_rgb)
        # GDINO text must be lowercase and end with a period
        q = text.lower().strip().rstrip('.') + '.'
        inputs = self.proc(images=img, text=q, return_tensors='pt').to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
        res = self.proc.post_process_grounded_object_detection(
            out, inputs['input_ids'], threshold=box_thr, text_threshold=text_thr,
            target_sizes=[img.size[::-1]])[0]
        if len(res['boxes']) == 0:
            return None
        # highest-scoring box
        i = int(res['scores'].argmax())
        x0, y0, x1, y1 = res['boxes'][i].tolist()
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1), (x0, y0, x1, y1), float(res['scores'][i]))


def pixel_to_grid(cx, cy, img_w, img_h, gw, gh):
    """Pixel center -> feature grid (gy, gx)."""
    gx = int(np.clip(cx / img_w * gw, 0, gw - 1))
    gy = int(np.clip(cy / img_h * gh, 0, gh - 1))
    return gy, gx
