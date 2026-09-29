"""AgiBot wmap-only gate arm: agi_joint_config without copy-paste (p_aug=0) and L_id, so the
motionauto dual-arm weightmap is the only change.
"""

from eveworld.agibot.agi_joint_config import config as _base

import copy
import os

config = copy.deepcopy(_base)
config['project_dir'] = (f"{os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))}/eve_v2_outputs/"
                         'agibot_ewm_apre/experiments')
_t = config['dataloaders']['train']['transform']
_t['p_aug'] = 0.0
_t['p_aug_state'] = 0.0
config['models']['t4g_lambdas'] = '0.0,0.0'
config['train']['max_steps'] = 150
config['train']['checkpoint_total_limit'] = 4
