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

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TesnroboardLoggerHook"),
        dict(type="MMDetWandbHook", by_epoch=False,
            init_kwargs={'entity': wandb_entity,
                         'project': project_name,}),])
                        #  'config': cfg_dict

# model config
model = dict(
    type='CTDetection', #name of detector
    backbone = dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)),
    
    neck = dict(
        type='OSA',  ###remove unused parameters 
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_name='BN',
        target_idx = 0,
        bn_requires_grad=False
    ),

    rpn_head = dict(

    ),
    roi_head = dict(

    ), 
)
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
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', color_type='grayscale'), #1 Channel IR Image
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
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
num_epochs = 20
runner = dict(
    type = "EpochBasedRunner", max_epoch=num_epochs)

lr_config = dict(
    policy='step', warmup='linear', warmup_ratio=0.001, step=[8, 11])

optimizer = dict(
    type = 'AdamW',
    lr=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone' : dict(lr_mult=0.25)
        }
    ),
    weight_decay=0.01
)

evaluation = dict(interval=1, metric='bbox')
load_from = None
resume_from = None