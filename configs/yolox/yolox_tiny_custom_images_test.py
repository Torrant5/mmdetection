_base_ = './yolox_tiny_8xb8-300e_coco_clearml.py'

# Custom dataset (images variant) for evaluation with pretrained YOLOX-Tiny

dataset_type = 'CocoDataset'
backend_args = None

# Point to the split directories built by .notes/py/build_dataset_pipeline.py
custom_root = 'data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/splits/images/'

# Keep 80 classes as in COCO; annotations actually include only {1,33}
model = dict(bbox_head=dict(num_classes=80))

img_scale = (640, 640)

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]

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
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=custom_root + 'test/instances_test.json',
    metric='bbox',
    backend_args=backend_args,
)

test_evaluator = val_evaluator

