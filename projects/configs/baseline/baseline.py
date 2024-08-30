# baseline config file
_base_ = [
    '../../../mmdetection/configs/_base_/datasets/coco_detection.py'
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
        
    )
    neck = dict(

    )

    rpn_head = dict(

    )
    roi_head = dict(

    )   
)
# dataset config
dataset_type = 'HanhwaIRDataset'
data_root = 'data/IRData/'
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
    train = dict(type='HanhwaIRDataset',
                 data_root=data_root,
                 ann_file=data_root+'annotations/train.json',
                 img_prefix=data_root +'train/',
                 pipeline=train_pipeline,
                 classes=class_names
                 ),
    val = dict(type='HanhwaIRDataset',
                 data_root=data_root,
                 ann_file=data_root+'annotations/val.json',
                 img_prefix=data_root +'val/',
                 pipeline=test_pipeline,
                 classes=class_names
                 ),
    test = dict(type='HanhwaIRDataset',
                 data_root=data_root,
                 ann_file=data_root+'annotations/val.json',
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
            'backbone' : dict(lr_mult:0.25)
        }
    )
    weight_decay=0.01
)

evaluation = dict(interval=1, metric='bbox')
load_from = None
resume_from = None