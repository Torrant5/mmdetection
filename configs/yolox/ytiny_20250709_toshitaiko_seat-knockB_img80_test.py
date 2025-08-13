_base_ = './yolox_tiny_8xb8-300e_coco_clearml.py'

dataset_type = 'CocoDataset'
backend_args = None

custom_root = 'data/20250709_YokohamaStadium_ToshiTaiko_seat-knock_B_200/splits/images/'

model = dict(bbox_head=dict(num_classes=80))

# Classes as defined in the custom annotations (underscored names)
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

# Enable built-in visualization hook to save test predictions
default_hooks = dict(
    visualization=dict(
        type='DetVisualizationHook', draw=True, interval=1, score_thr=0.3,
        test_out_dir='vis_test')
)
