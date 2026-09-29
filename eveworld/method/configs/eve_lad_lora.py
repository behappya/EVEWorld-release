# EVE LAD-LoRA training-time anti-laziness config (plan27 §20 main line).
# Derived from the physlatent adapter; changes:
#   - runner -> EveLadLoraTrainer
#   - train_mode='lora' (frozen backbone, LoRA only, targets to_{q,k,v,out}.0, rank 64)
#   - physics_latent.enabled=False (no physics tokens here, only a frozen-LAD regularizer)
#   - new models.lad_lora: frozen LAD ckpt + regularizer weight
# Main loss is still EDM denoising (protects image quality); lad_reg is the sigma-gated
# soft-top-k transition_error regularizer.
import copy
import os
from pathlib import Path
from eveworld.alternatives.physlatent.configs.gr1_physlatent_adapter import config as _base

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

config = copy.deepcopy(_base)
config['runners'] = ['eveworld.EveLadLoraTrainer']
config['launch']['deepspeed_config']['deepspeed_config_file'] = str(
    Path(__file__).with_name('deepspeed_zero2_clip.json').resolve()
)
# project_dir sets the checkpoint output path (trainer.py: cls(project_dir=config.project_dir,...)).
# Serial ablation w=0.1/0.0/0.2 sharing one project_dir would overwrite ckpts
# -> isolate per W_LAD+SEED.
# (W_LAD/SEED track w_lad; the launch trailing positionals export them inside the
# kjob, so they are readable when the config runs.)
_w_lad_dir = os.environ.get('W_LAD', '0.1')
_seed_dir = os.environ.get('SEED', '6666')
config['project_dir'] = (
    f'{GAGI}/giga_world_0_outputs/eve/'
    f'experiments_lad_lora_w{_w_lad_dir}_seed{_seed_dir}'
)

# LoRA fine-tuning (frozen backbone)
config['models']['train_mode'] = 'lora'
config['models']['lora_rank'] = 64
config['optimizers']['lr'] = float(os.environ.get('LAD_LORA_LR', str(config['optimizers']['lr'])))

# disable the physics latent conditioning path (only a frozen LAD regularizer is needed, no physics tokens)
config['models']['physics_latent']['enabled'] = False
config['models']['physics_latent']['train_backbone'] = False

# LAD-LoRA anti-laziness regularizer (EveLadLoraTrainer.get_models reads model_config['lad_lora']).
# w_lad comes from an env var for ablation (W_LAD=0.0 = pure LoRA control, rules out
# "just more LoRA params"); sweep 0.05/0.1/0.2 for the quality-fidelity Pareto.
config['models']['lad_lora'] = dict(
    w_lad=float(os.environ.get('W_LAD', '0.1')),  # regularizer weight; 0 = control
    topk=int(os.environ.get('LAD_TOPK', '3')),    # soft-top-k over the 3 largest illegal-jump spikes
    tau=float(os.environ.get('LAD_TAU', '0.5')),  # soft-max temperature (smaller -> closer to hard max)
    sigma_max=float(os.environ.get('LAD_SIGMA_MAX', '0.0')),  # >0 hard gating; default soft 1/(1+sigma)
    # w_lad is the target LAD/EDM gradient ratio at the shared x0; this cap only prevents
    # blow-up when the LAD gradient is near 0.
    balance_max_scale=float(os.environ.get('LAD_BALANCE_MAX_SCALE', '1.0')),
    lam_ckpt=os.environ.get('LAM_CKPT', f'{GAGI}/eve_outputs/lam/lam_gr1.pt'),
)

# Diagnostics default to 50 steps, checkpoint every 5. The launcher/runtime writer
# may override, but keep a safe default here.
config['train']['max_steps'] = 50
config['train']['checkpoint_interval'] = 5
config['train']['checkpoint_total_limit'] = -1
config['train']['with_ema'] = False
# Under DeepSpeed max_grad_norm is only an expected value / startup assertion;
# the real clipping is done by the dedicated DS JSON above.
config['train']['max_grad_norm'] = float(os.environ.get('MAX_GRAD_NORM', '1.0'))
# Resume disabled by default, so different weights or a diverged state cannot silently continue.
config['train']['resume'] = os.environ.get('RESUME', '0') == '1'
