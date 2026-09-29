"""Strict CP-only arm: Copy-Paste correction with uniform diffusion loss."""

from eveworld.pipeline.igr.config import config as base_config
from eveworld.pipeline.ablation.strict_ablation_common import strict_config


config = strict_config(base_config, "cp_only")
config["dataloaders"]["train"]["transform"].update(p_aug=0.5, strict_wmap=False)
