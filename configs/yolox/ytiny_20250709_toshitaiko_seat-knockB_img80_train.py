_base_ = './yolox_tiny_8xb8-300e_coco_clearml.py'

dataset_type = 'CocoDataset'
backend_args = None

custom_root = 'data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/splits/images/'

model = dict(bbox_head=dict(num_classes=80))

classes = (
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic_light','fire_hydrant','stop_sign','parking_meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports_ball','kite','baseball_bat','baseball_glove','skateboard','surfboard','tennis_racket','bottle',
    'wine_glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot_dog','pizza','donut','cake','chair','couch','potted_plant','bed',
    'dining_table','toilet','tv','laptop','mouse','remote','keyboard','cell_phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors','teddy_bear','hair_drier','toothbrush'
)

img_scale = (640, 640)

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
    dict(type='RandomAffine', scaling_ratio_range=(0.1, 2), border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='MixUp', img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
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
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=8,
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

# Shorten schedule for quick fine-tuning (override via CLI)
max_epochs = 20
num_last_epochs = 5
interval = 5

train_cfg = dict(max_epochs=max_epochs, val_interval=interval)

param_scheduler = [
    dict(type='mmdet.QuadraticWarmupLR', by_epoch=True, begin=0, end=1, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=1e-4, begin=1, T_max=max_epochs - num_last_epochs,
         end=max_epochs - num_last_epochs, by_epoch=True, convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=max_epochs - num_last_epochs, end=max_epochs),
]

default_hooks = dict(checkpoint=dict(interval=interval, max_keep_ckpts=3))

# Also enable drawing during val/test; test_out_dir is used by tools/test.py
default_hooks.update(dict(
    visualization=dict(
        type='DetVisualizationHook', draw=True, interval=1, score_thr=0.3,
        test_out_dir='vis_test')
))
