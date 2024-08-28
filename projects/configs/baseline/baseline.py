baseline config file
_base_ = [ ... ]

project = 'baseline'
project_name = 'baseline'

backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin=True
plugin_dir='projects/mmdet_plugin/'

... writing ..