"""Track4Gen 式对应监督训练 config (42 号 §3)。

kjob 尾随 KEY=VALUE / env 覆盖同名默认值; eveworld/pipeline/tia/launch.sh 才是最终事实来源。
"""
import os

GAGI = os.environ.get('GAGI_ROOT', os.path.expanduser('~/gagi'))

ANNO_DIR = f'{GAGI}/eve_v2_outputs/track4gen_probe/t4g_anno'
IDX2VID = ANNO_DIR + '/_packidx2vid.json'
ROUND0_EMA_TRANSFORMER = f'{GAGI}/eve_v2_outputs/anchor_models/round0_ema_st/transformer'

config = dict(
    runners=['eveworld.pipeline.tia.trainer.T4GCorrTrainer'],
    project_dir=f'{GAGI}/eve_v2_outputs/t4g_corr/experiments',
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
                type='T4GCorrTransform',
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
                idx2vid_path=IDX2VID,
                anno_dir=ANNO_DIR,
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
        # t4g 对应 loss 超参 (43 号实证版; env 可再覆盖)
        t4g_id_block='block22',      # L_id 甜点层 (E4 EPE0.55)
        t4g_change_block='block25',  # L_change 甜点层 (E7 gap+0.45)
        t4g_sigma_lo=0.2,
        t4g_sigma_hi=0.5,
        t4g_warmup=20,
        t4g_lambdas='0.5,0.4',       # (id, change) 两道
        t4g_tau=0.07,
        t4g_tol=0.2,                 # 静止-变化容忍带 (E7 静止地板0.80)
        t4g_anno_dir=ANNO_DIR,
        t4g_idx2vid=IDX2VID,
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
        max_steps=50,                 # 探针档 (launcher MAX_STEPS 覆盖)
        gradient_accumulation_steps=8,
        mixed_precision='bf16',
        checkpoint_interval=50,
        checkpoint_total_limit=8,
        checkpoint_start_step=0,      # 自定义键: step<该值不存档 (探针=0 存全部)
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=['TransformerBlock'],
    ),
)
