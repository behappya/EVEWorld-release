"""AgiBot dual-arm full recipe: pretrain base + skill-conditioned copy-paste + dual-arm CWM + L_id.

640x480 / latent 30x40, so the launcher must export T4G_W_LAT=40 T4G_WPIX=640.
"""

import os

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))
PROBE = GAGI + '/eve_v2_outputs/agibot_t4g_probe'
ANNO_DIR = PROBE + '/t4g_anno'
ASSETS_DIR = PROBE + '/aug_assets'
WMAP_DIR = PROBE + '/weightmap_cache_motionauto'
PRETRAIN = GAGI + '/giga_world_0_video_pretrain'

config = dict(
    runners=['eveworld.agibot.agi_aug_trainer.AgiJointTrainer'],
    project_dir=GAGI + '/eve_v2_outputs/agibot_ewm_apre/experiments',
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[GAGI + '/agibot_ewm_packed'],
            batch_size_per_gpu=1,
            num_workers=6,
            transform=dict(
                type='AgiAugTransform',
                num_frames=93,
                height=480,
                width=640,
                fps=16,
                image_cfg=dict(
                    mask_generator=dict(max_ref_frames=1, start=1, factor=4),
                ),
                idx2vid_path=ANNO_DIR + '/_packidx2vid.json',
                anno_dir=ANNO_DIR,
                assets_dir=ASSETS_DIR,
                wmap_dir=WMAP_DIR,
                wmap_values='1.0,3.0',
                strict_mapping=True,
                strict_assets=True,
                strict_wmap=True,
                p_aug=0.35,
                p_aug_state=0.15,
                zone_bias='0.5,0.2,0.3',
                seed=20260728,
                p_corridor=0.0,
                p_alien=0.0,
            ),
            sampler=dict(type='DefaultSampler', shuffle=True),
        ),
    ),
    models=dict(
        vae_model_path=PRETRAIN + '/vae',
        transformer_model_path=PRETRAIN + '/transformer',
        t4g_id_block='block22',            # placeholder; T4G_ID_BLOCK=<Phase5 result> overrides it
        t4g_change_block='block25',
        t4g_sigma_lo=0.2,
        t4g_sigma_hi=0.5,
        t4g_warmup=20,
        t4g_lambdas='0.5,0.0',
        t4g_tau=0.07,
        t4g_tol=0.2,
        t4g_anno_dir=ANNO_DIR,
        t4g_idx2vid=ANNO_DIR + '/_packidx2vid.json',
        t4g_expected_samples=777,
        t4g_wmap_dir=WMAP_DIR,
        t4g_w_paste=3.0,
        t4g_region_levels='1.0,3.0',
        t4g_region_names='background,marked',
    ),
    optimizers=dict(type='CAME8Bit', lr=2 ** (-14.5)),
    schedulers=dict(type='ConstantScheduler'),
    train=dict(
        resume=True,
        max_steps=300,
        gradient_accumulation_steps=8,
        mixed_precision='bf16',
        checkpoint_interval=50,
        checkpoint_total_limit=8,
        checkpoint_start_step=0,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['TransformerBlock'],
        seed=42,
    ),
)
