_base_ = ['./_base_/default_runtime.py']

# Dataset configuration
dataset_type = 'CocoDataset'
classes = ('Normal_LS', 'Abnormal_LS')
data_root = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/'

# Training pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(640, 480), (1333, 800)],
        keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# Testing/validation pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# DataLoader configuration
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'train/train.json',
        data_prefix=dict(img=data_root + 'train/images/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler')
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'valid/valid.json',
        data_prefix=dict(img=data_root + 'valid/images/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file=data_root + 'test/test.json',
        data_prefix=dict(img=data_root + 'test/images/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline
    ),
    sampler=dict(type='DefaultSampler', shuffle=False)
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/valid.json',
    metric='bbox',
    format_only=False
)

#test_evaluator = val_evaluator
test_evaluator = dict(
    ann_file='/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/test/test.json',  # ✅ Correct test JSON
    format_only=False,
    metric='bbox',
    type='CocoMetric'
)


# Model configuration
num_stages = 6
num_proposals = 100

model = dict(
    type='SparseRCNN',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4
    ),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256
    ),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=2,  # 👈 Your number of classes
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')
                ),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0
                ),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1.0, 1.0]
                )
            )
            for _ in range(num_stages)
        ]
    ),
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ]
                ),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1
            )
            for _ in range(num_stages)
        ]
    ),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals))
)

# Optimizer and learning rate scheduler
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
       # lr=2.5e-5,
        lr=1e-4,
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2)
)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1
    )
]

# Training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    early_stopping=dict(
        type='EarlyStoppingHook',
        patience=10,
        monitor='coco/bbox_mAP'
    )
)

# Environment configuration
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl')
)
