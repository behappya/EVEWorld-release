"""Reproducible A-pre configuration with the correspondence layer at block 23."""

from eveworld.pipeline.train.recipes.cfg_repro_seed42_config import config as _base
import copy
import os

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

config = copy.deepcopy(_base)
config["models"]["t4g_id_block"] = "block23"
config["project_dir"] = \
    f"{GAGI}/eve_v2_outputs/t4g_cfg_repro_seed42_b23_s300/experiments"
