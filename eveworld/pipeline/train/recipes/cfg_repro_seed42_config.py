"""Reproducible A-pre configuration for the six-point CFG sweep."""

from eveworld.pipeline.train.recipes.apre_noaug_config import config as _base
import copy
import os

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

config = copy.deepcopy(_base)
config["dataloaders"]["train"]["transform"]["p_aug"] = 0.5
config["project_dir"] = \
    f"{GAGI}/eve_v2_outputs/t4g_cfg_repro_seed42_s300/experiments"
config["train"]["seed"] = 42
config["train"]["max_steps"] = 300
config["train"]["checkpoint_interval"] = 50
config["train"]["checkpoint_total_limit"] = 8

