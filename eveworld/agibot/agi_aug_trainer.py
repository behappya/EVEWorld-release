#!/usr/bin/env python3
"""AgiBot skill-conditioned augmentation transform + trainer shell.

STATE skills get pure-background pasting only (p_aug_state=0.15): their end state cannot be
defined by pasting a patch, so object pasting stays off.
"""
from __future__ import annotations

import json
import os

from giga_train import TRANSFORMS

from ..pipeline.igr.trainer import T4GAugTransform
from ..pipeline.train.joint.trainer import T4GJointTrainer

SKILL_ZONE_BIAS = {
    'Pick': (0.0, 0.5, 0.5),
    'Place': (0.5, 0.2, 0.3),
    'HandOver': (0.5, 0.2, 0.3),
    'Insert': (0.5, 0.2, 0.3),
    'Push': (0.2, 0.4, 0.4),
    'Pull': (0.2, 0.4, 0.4),
}


@TRANSFORMS.register
class AgiAugTransform(T4GAugTransform):

    def __init__(self, *args, p_aug_state=0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_aug_state = float(p_aug_state)
        self._skill_cache = {}

    def _skill_meta(self, vid):
        if vid not in self._skill_cache:
            fp = os.path.join(self.anno_dir, f'{vid}.json')
            meta = None
            if os.path.exists(fp):
                a = json.load(open(fp))
                meta = {'skill': a.get('skill'), 'category': a.get('category', 'TRANSFER')}
            self._skill_cache[vid] = meta
        return self._skill_cache[vid]

    def __call__(self, data_dict):
        di = data_dict.get('data_index', None)
        vid = self.idx2vid.get(int(di)) if di is not None else None
        meta = self._skill_meta(str(vid)) if vid is not None else None
        # single-threaded inside a dataloader worker, so a temporary self mutation is safe
        orig_p, orig_zb = self.p_aug, self.zone_bias
        if meta is not None:
            if meta['category'] == 'STATE':
                self.p_aug = self.p_aug_state
                self.zone_bias = (0.0, 0.0, 1.0)
            else:
                self.zone_bias = SKILL_ZONE_BIAS.get(meta['skill'], orig_zb)
        try:
            return super().__call__(data_dict)
        finally:
            self.p_aug, self.zone_bias = orig_p, orig_zb


class AgiJointTrainer(T4GJointTrainer):
    """Bit-identical to T4GJointTrainer; exists only so runners import this module and register."""
