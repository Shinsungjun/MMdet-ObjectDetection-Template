# baseline config file
_base_ = [
    '../../../mmdetection/configs/_base_/datasets/coco_detection.py',
    '../../../mmdetection/configs/_base_/schedules/schedule_1x.py',
    '../../../mmdetection/configs/_base_/default_runtime.py'
]
# (Above) Override on base config

# project config
project = 'baseline'
project_name = 'baseline'

wandb_entity = 'holyjoon'

backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet_plugin/'
max_epochs = 300
num_last_epochs = 15
resume_from = None
interval = 10
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),])
                        #  'config': cfg_dict
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=num_last_epochs,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=num_last_epochs,
        interval=interval,
        priority=48),
    dict(
        type='ExpMomentumEMAHook',
        resume_from=resume_from,
        momentum=0.0001,
        priority=49)
]
# model config
img_scale = (480, 640)
model = dict(
    type='YOLOX',
    input_size=img_scale,
    random_size_range=(15, 25),
    random_size_interval=10,
    backbone=dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.5),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1),
    bbox_head=dict(
        type='CustomYOLOXHead', num_classes=8, in_channels=128, feat_channels=128),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    # In order to align the source code, the threshold of the val phase is
    # 0.01, and the threshold of the test phase is 0.001.
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)))

# dataset config
dataset_type = 'HanhwaIRDataset'
data_root = '/ws/HanhwaIRChallenge/MMdet-ObjectDetection/data/IRData/' #for ipynb
# data_root = './data/IRData/'
class_names = ['person', 'car', 'truck', 'bus', 'bicycle', 'bike', 'extra_vehicle', 'dog']
num_gpus = 1
batch_size = 4
# num_iters_per_epoch = 
img_norm_cfg = None #For IR Image

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'), #1 Channel IR Image
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='OneChannelImgFormatBundle'),
    dict(type='CollectIR', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'), #1 Channel IR Image
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640,480),
        flip=False,
        transforms=[
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='OneChannelImgFormatBundle'),
    dict(type='CollectIR', keys=['img']),
    ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    shuffle=True, #only in training
    train = dict(type='HanhwaIRDataset',
                 data_root=data_root,
                 ann_file='annotations/train.json',
                 img_prefix=data_root +'train/',
                 pipeline=train_pipeline,
                 classes=class_names
                 ),
    val = dict(type='HanhwaIRDataset',
                 data_root=data_root,
                 ann_file='annotations/val.json',
                 img_prefix=data_root +'val/',
                 pipeline=test_pipeline,
                 classes=class_names
                 ),
    test = dict(type='HanhwaIRDataset',
                 data_root=data_root,
                 ann_file='annotations/val.json',
                 img_prefix=data_root +'val/',
                 pipeline=test_pipeline,
                 classes=class_names
                 ),)

# train/inference config
runner = dict(
    type = "EpochBasedRunner", max_epochs=max_epochs)

lr_config = dict(
    _delete_=True,
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=num_last_epochs,
    min_lr_ratio=0.05)

optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)

evaluation = dict(interval=1, metric='bbox')
load_from = None
resume_from = None