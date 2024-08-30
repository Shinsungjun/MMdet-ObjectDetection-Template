# baseline config file
_base_ = [
    '../../../mmdetection/configs/_base_/datasets/coco_detection.py'
]
# (Above) Override on base config

# project config
project = 'baseline'
project_name = 'baseline'

backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet_plugin/'
# model config

# dataset config
dataset_type = 'HanhwaIRDataset'
data_root = 'data/IRData/'
class_name = ['person', 'car', 'truck', 'bus', 'bicycle', 'bike', 'extra_vehicle', 'dog']
num_gpus = 1
batch_size = 4
# num_iters_per_epoch = 
img_norm_cfg = None #For IR Image
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

train_pipeline = [
    dict(type='LoadIRImageFromFile'),
    dict(type='LoadHanhwaAnnotations', with_bbox=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadIRImageFromFile'),
    dict(type='RandomFlip'),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train = dict(),
    val = dict(),
    test = dict( ))

# train/inference config
num_epochs = 20
runner = dict(
    type = "EpochBasedRunner", max_epoch=num_epochs)

lr_config = dict(
    policy='step', warmup='linear', warmup_ratio=0.001, step=[8, 11])

evaluation = dict(interval=1, metric='bbox')
load_from = None
resume_from = None