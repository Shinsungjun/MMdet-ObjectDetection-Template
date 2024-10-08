# baseline config file
_base_ = [
    '../../../mmdetection/configs/_base_/datasets/coco_detection.py',
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

# model config
img_scale = (480, 640)
model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=8,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
# dataset config
dataset_type = 'HanhwaIRDataset'
data_root = '/ws/HanhwaIRChallenge/MMdet-ObjectDetection/data/IRData/' #for ipynb
# data_root = './data/IRData/'
class_names = ['person', 'car', 'truck', 'bus', 'bicycle', 'bike', 'extra_vehicle', 'dog']
num_gpus = 1
batch_size = 8
# num_iters_per_epoch = 
img_norm_cfg = None #For IR Image

train_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'), #1 Channel IR Image
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='OneChannelImgFormatBundle'),
    dict(type='CollectIR', keys=['img', 'gt_bboxes', 'gt_labels']), #1Channel IR -> 3Channel IR (just copy)
]

val_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'), #1 Channel IR Image
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640,480),
        flip=False,
        transforms=[
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='OneChannelImgFormatBundle'),
    dict(type='CollectIR', keys=['img']), #1Channel IR -> 3Channel IR (just copy)
    ])
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'), #1 Channel IR Image
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640,480),
        flip=False,
        transforms=[
    dict(type='OneChannelImgFormatBundle'),
    dict(type='CollectIR', keys=['img']), #1Channel IR -> 3Channel IR (just copy)
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
                 pipeline=val_pipeline,
                 classes=class_names
                 ),
    test = dict(type='HanhwaIRDataset',
                 data_root=data_root,
                 ann_file='annotations/test.json',
                 img_prefix=data_root +'test_open/',
                 pipeline=test_pipeline,
                 classes=class_names,
                 test_mode=True
                 ),)

# train/inference config
runner = dict(
    type = "EpochBasedRunner", max_epochs=max_epochs)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])
auto_scale_lr = dict(base_batch_size=8)
evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)

load_from = None
resume_from = None