"""Strict IGT arm: Copy-Paste plus CWM, with CIC disabled."""

from eveworld.pipeline.train.joint.config import config as base_config
from eveworld.pipeline.ablation.strict_ablation_common import strict_config, strict_joint_metadata


config = strict_config(base_config, "igt")
config["dataloaders"]["train"]["transform"]["p_aug"] = 0.5
config["models"]["t4g_lambdas"] = "0.0,0.0"
strict_joint_metadata(config)
