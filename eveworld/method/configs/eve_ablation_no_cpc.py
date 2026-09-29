# Ablation: CPC off (degenerates to a plain full-fine-tuning baseline)
from eveworld.method.configs.eve_causal_fullft import config
import copy
config = copy.deepcopy(config)
config['models']['causal']['w_cpc'] = 0.0
config['project_dir'] = config['project_dir'].replace('causal_fullft','ablation_no_cpc')
