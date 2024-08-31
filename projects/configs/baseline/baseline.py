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
        dict(type="TextLoggerHook"),])
                        #  'config': cfg_dict

# model config
model = dict(
    type='DETR',
    backbone=dict(
        type='ResNet',
        in_channels=1, #IR Data
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='DETRHead',
        num_classes=8, #IRData
        in_channels=2048,
        transformer=dict(
            type='Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))
    
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
    dict(type='DefaultFormatBundle'),
    dict(type='CollectIR', keys=['img', 'gt_bboxes', 'gt_labels']),
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
num_epochs = 20
runner = dict(
    type = "EpochBasedRunner", max_epochs=num_epochs)

lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr_ratio=1e-2,
    warmup='linear',
    warmup_ratio=1e-3,
    warmup_iters=10 * 1252,
    warmup_by_epoch=False)

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
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
evaluation = dict(interval=1, metric='bbox')
load_from = None
resume_from = None