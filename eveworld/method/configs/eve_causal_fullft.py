# EVE causal process fidelity: full fine-tuning config.
# Derived from the physlatent adapter config (reusing its valid schema); only changes:
#   - runner -> EveCausalTrainer
#   - physics_latent.train_backbone=True (full fine-tuning; the prototype's main failure cause)
#   - physics aux losses all 0 (no weak-label auxiliary heads)
#   - new models.causal (EVE causal loss weights, read by EveCausalTrainer.get_models)
import copy
import os
from eveworld.alternatives.physlatent.configs.gr1_physlatent_adapter import config as _base

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

config = copy.deepcopy(_base)
config['runners'] = ['eveworld.EveCausalTrainer']
config['project_dir'] = f'{GAGI}/giga_world_0_outputs/eve/experiments_causal_fullft'

# full fine-tuning + physics auxiliary heads off
pl = config['models']['physics_latent']
pl['train_backbone'] = True
pl['loss_weights'] = dict(state=0.0, goal=0.0, contact=0.0, trajectory=0.0)

# EVE causal losses (EveCausalTrainer reads model_config['causal'])
config['models']['causal'] = dict(
    w_cpc=0.5,                       # counterfactual process comparison (C)
    neg_kinds=['teleport', 'excision', 'freeze_jump'],
    margin=0.1,
    use_random_neg=False,            # set True for the ablation
    # w_dyn/w_prog left 0: validate the CPC backbone first; enable LAM/progress gating after P0
    w_dyn=0.0,
    w_prog=0.0,
    lam_ckpt=os.environ.get('LAM_CKPT', f'{GAGI}/eve_outputs/lam/lam_pretrained.pt'),
)

# steps etc. follow base; override with the kjob MAX_STEPS env var
config['train']['max_steps'] = 2000
config['train']['checkpoint_interval'] = 200
