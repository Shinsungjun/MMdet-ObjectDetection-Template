# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by SungJun Shin
# ---------------------------------------------

import numpy as np
import torch
import random
import math
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet.datasets import CustomDataset
from mmdet.datasets.api_wrappers import COCO, COCOEval
from mmdet.core import eval_recalls

@DATASETS.register_module()
class HanhwaIRDataset(CustomDataset):
    """
    HanHwa IR Object Detection Dataset (COCO Format)
    diff : image id is string (coco's image id is int)
    """
    
    CLASSES = ()
    
    PALETTE = []
    
    def load_annotations(self, ann_file):
        """_summary_

        Args:
            ann_file (_type_): _description_
        """
        
        self.coco = COCO(ann_file)
        
        data_infos = []
        # ~~
        
        return data_infos
    
    def get_ann_info(self, idx):
        """_summary_
            get annotation info with string image id
        Args:
            idx (_type_): _description_
        """
        pass
    
    def get_cat_ids(self, idx):
        pass
    
    def _filter_imgs(self, min_size=32):
        pass
    
    def _parse_ann_info(self, img_info, ann_info):
        pass
    
    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]
        
    def _proposal2json(self, results):
        pass
    
    def _det2json(self, results):
        pass
    
    