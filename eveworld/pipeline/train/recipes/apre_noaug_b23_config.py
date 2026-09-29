"""A-pre-noaug with the correspondence layer at block 23 instead of block 22."""

from eveworld.pipeline.train.recipes.apre_noaug_config import config as _base
import copy
import os

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

config = copy.deepcopy(_base)
config["models"]["t4g_id_block"] = "block23"
config["project_dir"] = f"{GAGI}/eve_v2_outputs/t4g_joint_wmapA_pre_noaug_b23/experiments"
