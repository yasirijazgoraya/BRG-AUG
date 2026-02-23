_base_ = '/home/yasir/yasir_mnt/external3/object_detectors/mmdetection/checkpoints/rtmdet_s_8xb32-300e_coco.py'

# Dataset metadata
metainfo = {
    'classes': ('Normal_LS', 'Abnormal_LS'),
    'palette': [(255, 0, 0), (0, 255, 0)]
}

# Common dataset root
base_data_root = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/'

# Model configuration
model = dict(
    backbone=dict(
        deepen_factor=0.167,
        widen_factor=0.375
    ),
    neck=dict(in_channels=[96, 192, 384], out_channels=96, num_csp_blocks=1),
    bbox_head=dict(in_channels=96, feat_channels=96, exp_on_reg=False, num_classes=2)
)

# Optimizer and learning schedule
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0005,
        weight_decay=0.05
    ),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True)
)

# Learning rate scheduler
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=0, end=100, T_max=100, eta_min=1e-6)
]

# Data augmentation pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0, max_cached_images=20, random_pop=False),
    dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='CachedMixUp', img_scale=(640, 640), ratio_range=(1.0, 1.0), max_cached_images=10, random_pop=False, pad_val=(114, 114, 114), prob=0.5),
    dict(type='PackDetInputs')
]

# DataLoaders
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='',
        ann_file=base_data_root + 'train/train.json',
        data_prefix=dict(img=base_data_root + 'train/images'),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='',
        ann_file=base_data_root + 'valid/valid.json',
        data_prefix=dict(img=base_data_root + 'valid/images'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='',
        ann_file=base_data_root + 'test/test.json',
        data_prefix=dict(img=base_data_root + 'test/images'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

# Evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=base_data_root + 'valid/valid.json',
    metric='bbox'
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=base_data_root + 'test/test.json',
    metric='bbox'
)

# Runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')



default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'),  # ✅ correct hook name
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='coco/bbox_mAP',
        patience=10
    )
)






resume = True



default_scope = 'mmdet'
