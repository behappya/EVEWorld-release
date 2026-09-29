# Ablation: random-permutation negatives instead of causal negatives (shows the causal signal is what matters)
from eveworld.method.configs.eve_causal_fullft import config
import copy
config = copy.deepcopy(config)
config['models']['causal']['use_random_neg'] = True
config['project_dir'] = config['project_dir'].replace('causal_fullft','ablation_random_neg')
