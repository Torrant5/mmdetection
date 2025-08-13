"""
Fine-tuning configuration for YOLOX-Tiny with 2 classes (person, sports ball).
Optimized for small dataset with transferred weights from COCO.
"""

_base_ = './yolox_tiny_8xb8-300e_coco_clearml.py'

# Use transferred weights from 80-class model (person and sports ball preserved)
load_from = 'weights/yolox_tiny_2class_transferred_new.pth'

dataset_type = 'CocoDataset'
backend_args = None

custom_root = 'data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/splits/images2/'

model = dict(
    bbox_head=dict(num_classes=2),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65))
)

classes = ('person', 'sports ball')

img_scale = (3040, 3040)  # High resolution for small object detection

# Minimal augmentation for fine-tuning (compatible with YOLOX pipeline)
train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='RandomAffine', scaling_ratio_range=(0.9, 1.1), border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='MixUp', img_scale=img_scale, ratio_range=(1.0, 1.0), pad_val=114.0),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

# Dataset with MultiImageMixDataset wrapper (required for YOLOX)
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=custom_root,
        ann_file='train/instances_train.json',
        data_prefix=dict(img='train/'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True),
        ],
        metainfo=dict(classes=classes),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        backend_args=backend_args,
    ),
    pipeline=train_pipeline,
)

train_dataloader = dict(
    batch_size=1,  # Small batch size for high resolution
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=custom_root,
        ann_file='test/instances_test.json',
        data_prefix=dict(img='test/'),
        test_mode=True,
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=custom_root + 'test/instances_test.json',
    metric='bbox',
    classwise=True,
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# Training configuration
max_epochs = 30
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,  # Validate every epoch
)

# Very conservative learning rate for fine-tuning
base_lr = 0.00001  # Conservative LR for fine-tuning: 1/1000 of default (0.01)

# Simple optimizer without complex weight decay settings
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.9,
        weight_decay=0.0001,  # Reduced weight decay
        nesterov=True
    ),
    clip_grad=None  # No gradient clipping
)

# Gentle learning rate schedule
param_scheduler = [
    # Constant learning rate for first 5 epochs
    dict(
        type='ConstantLR',
        factor=1.0,
        begin=0,
        end=5,
        by_epoch=True
    ),
    # Slow cosine decay
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs - 5,
        eta_min=base_lr * 0.1,
        begin=5,
        end=max_epochs,
        by_epoch=True
    )
]

# Hooks configuration
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        max_keep_ckpts=5,
        save_best='coco/bbox_mAP',
        rule='greater'
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='DetVisualizationHook',
        draw=True,
        interval=1,
        score_thr=0.01,  # Low threshold to see all detections
        test_out_dir='vis_test'
    )
)

# Custom hooks for fine-tuning
custom_hooks = [
    # Disable Mosaic/MixUp from the beginning
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=30,  # All epochs without Mosaic/MixUp
        priority=48
    ),
    dict(type='SyncNormHook', priority=48),
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,  # Slower EMA update
        update_buffers=True,
        priority=49
    )
]

# Experiment name
exp_name = 'yolox_tiny_2class_finetune'

# Log level
log_level = 'INFO'

# Resume from checkpoint if exists
resume = False

# Disable automatic learning rate scaling
auto_scale_lr = dict(enable=False)