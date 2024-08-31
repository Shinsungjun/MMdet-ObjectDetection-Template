# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by SungJun Shin
# ---------------------------------------------

import torch
import torch.nn.functional as F
import numpy as np
import mmcv

from mmcv.runner import force_fp32, auto_fp16, get_dist_info
from mmdet.models import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector

from mmdet.models import builder

@DETECTORS.register_module()
class CTDetection(TwoStageDetector):
    """
    """
    
    def __init__(self, 
                 backbone, 
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(CTDetection, self).__init__(backbone=backbone,
                                          neck=neck,
                                          rpn_head=rpn_head,
                                          roi_head=roi_head,
                                          train_cfg=train_cfg,
                                          test_cfg=test_cfg,
                                          pretrained=pretrained,
                                          init_cfg=init_cfg)
        
        def extract_feat(self, img):
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            return x

        def forward_train(self,
                          img,
                          img_metas,
                          gt_bboxes,
                          gt_labels,
                          gt_bboxes_ignore=None,
                          gt_masks=None,
                          proposal=None,
                          **data):
            x = self.extract_feat(img)

            losses = dict()

            if self.with_rpn: #center location
                ct_loss, outs_rpn = self.rpn_head(x)
                losses.update(ct_loss)

            else: #all location
                pass
            pred_bboxes, det_loss = self.rpn_head(x, outs_rpn, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **data)            
            losses.update(det_loss)

        def forward_test(self, img_metas, **data):
            return self.simple_test(img_metas, **data)

        def simple_test(self, img_metas, **data):
            img = data['img']
            x = self.extract_feat(img)
            losses = dict()

            if self.with_rpn: #center location
                ct_loss, outs_rpn = self.rpn_head(x)
                losses.update(ct_loss)

            else: #all location
                pass
            
            pred_bboxes, det_loss = self.rpn_head(x, outs_rpn, gt_bboxes, gt_labels, gt_bboxes_ignore, gt_masks, **data)            
            losses.update(det_loss)

            return pred_bboxes

        @auto_fp16(apply_to('img', ))
        def forward(self, return_loss=True, **data):
            if return_loss:
                return self.forward_train(**data)

            else:
                return self.forward_test(**data)


