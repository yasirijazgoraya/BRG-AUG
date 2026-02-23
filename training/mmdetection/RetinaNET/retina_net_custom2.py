_base_ = 'mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'

# Dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/yasir/yasir_mnt/external3/photorealistic_geometric/training_data_journal_paper/D3/'

metainfo = dict(
    classes=('Normal_LS', 'Abnormal_LS')
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/train.json',
        data_prefix=dict(img='train/images'),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='valid/valid.json',
        data_prefix=dict(img='valid/images'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/test.json',
        data_prefix=dict(img='test/images'),
        metainfo=metainfo,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='PackDetInputs')
        ]
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'valid/valid.json',
    metric='bbox',
    format_only=False
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/test.json',
    metric='bbox',
    format_only=False
)

# Model settings
model = dict(
    bbox_head=dict(
        num_classes=2  # Normal_LS, Abnormal_LS
    )
)

# Training schedule
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Checkpoint and logging
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1),
    logger=dict(type='LoggerHook', interval=50)
)

# Visualization
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer'
)

# Load pre-trained RetinaNet weights (for finetuning or inference)
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/retinanet/' \
            'retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
