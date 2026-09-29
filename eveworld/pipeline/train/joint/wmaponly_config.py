"""Joint training config: contract weightmap only (ablation: no L_id, no aug)
(stage 2 of arm 44, 92 GT clips, 200 steps)."""

import os

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

ANNO_DIR = f'{GAGI}/eve_v2_outputs/track4gen_probe/t4g_anno'
ASSETS_DIR = f'{GAGI}/eve_v2_outputs/track4gen_probe/aug_assets'
ROUND0_EMA_TRANSFORMER = f'{GAGI}/eve_v2_outputs/anchor_models/round0_ema_st/transformer'

config = dict(
    runners=['eveworld.pipeline.train.joint.trainer.T4GJointTrainer'],
    project_dir=f'{GAGI}/eve_v2_outputs/t4g_wmaponly/experiments',
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                f'{GAGI}/gr1_finetune_data/packed_data',
            ],
            batch_size_per_gpu=1,
            num_workers=6,
            transform=dict(
                type='T4GAugTransform',
                num_frames=93,
                height=480,
                width=768,
                fps=16,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
                idx2vid_path=ANNO_DIR + '/_packidx2vid.json',
                assets_dir=ASSETS_DIR,
                p_aug=0.0,                                     # ablation: aug off, clean data + weightmap only
                zone_bias='0.4,0.2,0.4',
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
        ),
    ),
    models=dict(
        vae_model_path=f'{GAGI}/giga_world_0_video_pretrain/vae',
        transformer_model_path=ROUND0_EMA_TRANSFORMER,
        # L_id (value validated by the corr probe)
        t4g_id_block='block22',
        t4g_change_block='block25',
        t4g_sigma_lo=0.2,
        t4g_sigma_hi=0.5,
        t4g_warmup=20,
        t4g_lambdas='0.0,0.0',          # ablation: L_id/L_change off, weightmap only
        t4g_tau=0.07,
        t4g_tol=0.2,
        t4g_anno_dir=ANNO_DIR,
        t4g_idx2vid=ANNO_DIR + '/_packidx2vid.json',
        # augmentation (E11-validated values)
        t4g_w_paste=4.0,
    ),
    optimizers=dict(
        type='CAME8Bit',
        lr=2 ** (-14.5),
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=True,
        max_steps=200,
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
    ),
)
