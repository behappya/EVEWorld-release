"""Joint training with a 3x paste-region weight instead of the historical 4x."""

from eveworld.pipeline.train.joint.config import config as _base
import copy
import os

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

config = copy.deepcopy(_base)
config["models"]["t4g_w_paste"] = 3.0
config["project_dir"] = \
    f"{GAGI}/eve_v2_outputs/t4g_joint_w_paste3/experiments"
