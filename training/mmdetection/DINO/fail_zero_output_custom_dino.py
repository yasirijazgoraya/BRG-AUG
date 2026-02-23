_base_ = [
    './_base_/datasets/coco_detection.py',
    './_base_/default_runtime.py'
]

#_base_ = [
 #   'mmdetection/configs/_base_/datasets/coco_detection.py',
  #  'mmdetection/configs/_base_/default_runtime.py'
#]


# Model (same as default DINO with minor change for num_classes)
model = dict(
    type='DINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20),
    bbox_head=dict(
        type='DINOHead',
        num_classes=2,  # ← your custom classes
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
    test_cfg=dict(max_per_img=300)
)

# Classes and dataset root
metainfo = {
    'classes': ('Normal_LS', 'Abnormal_LS'),
    'palette': [(255, 0, 0), (0, 255, 0)]
}

base_data_root = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/'

# Data augmentations (default from DINO paper)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

# Data loaders
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        ann_file=base_data_root + 'train/train.json',
        data_prefix=dict(img=base_data_root + 'train/images/'),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        ann_file=base_data_root + 'valid/valid.json',
        data_prefix=dict(img=base_data_root + 'valid/images/'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ])
)

#test_dataloader = val_dataloader

# Evaluators
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        ann_file=base_data_root + 'test/test.json',
        data_prefix=dict(img=base_data_root + 'test/images/'),
        pipeline=test_pipeline
    )
)


test_evaluator = dict(
    type='CocoMetric',
    ann_file=base_data_root + 'test/test.json',
    metric='bbox'
)


# Training schedule
max_epochs = 100  # ← default is 12, updated for your training
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)

# LR scheduler
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True, milestones=[90], gamma=0.1)
]

# Auto LR scaling
auto_scale_lr = dict(base_batch_size=16)

# Runtime
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
   # visualization=dict(type='VisualizationHook')
)

default_scope = 'mmdet'

# Save directory
work_dir = './work_dirs/dino_r50_custom'
