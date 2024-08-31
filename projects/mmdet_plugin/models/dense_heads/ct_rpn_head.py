# ------------------------------------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Sungjun Shin
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.models import HEADS, build_loss
#clip sigmoid?

@HEADS.register_module()
class CenterRPNHead(nn.Module)
    def __init__(self,
                 in_channels=256,
                 embed_dims=256,
                 sample_point=3000,
                 stride=4,
                 loss_centerness = dict(type='GaussianForcalLoss', reduction='mean'),
                 **kwargs):
        super(CenterRPNHead, self).__init__()

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.sample_point = sample_point
        self.stride = stride
        self.loss_centerness = build_loss(loss_centerness)

        self._init_layers()

    def _init_layers(self):
        self.conv = nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)

        self.center_pred = nn.Conv2d(self.embed_dims, 1, kernel_size=1)

        bias_init = bias_init_with_prob(0.01)
        nn.init_constant_(self.center_pred.bias, bias_init)

    def forward(self, **data):
        src = data['img_feats_det']
        bs, n, c, h, w= src.shape
        
        x = src.flatten(0, 1)
        cls_feat = self.conv(x)
        pred_centerness = self.center_pred(cls_feat) #BN, C, H, W
        centerness = pred_centerness.permute(0,2,3,1).reshape(bs, -1, 1)
        sample_weight = centerness.detach().sigmoid()
        
        #! topk index sampling
        _, topk_indexes = torch.topk(sample_weight, self.num_sample_points, dim=1)
        
        outs = {
                'centerness': pred_centerness,
                'topk_indexes':topk_indexes,
            }

        return outs
    
     @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_center,
             preds_dicts):
        pred_centerness = preds_dicts['centerness']
        loss_dict = dict()
        center_loss =  self.loss_single(gt_center, pred_centerness)
        loss_dict['center_loss'] = center_loss
        return loss_dict

    def loss_single(self, #! need to fix loss
                    gt_center,
                    pred_center):
        gt_centerness, pred_centerness = self.get_targets(gt_center, pred_center)
        # print("gt_centerness shape : ", gt_centerness.shape) #torch.Size([2, 6, 256, 704])
        pred_centerness = clip_sigmoid(pred_centerness)
        centerness_loss = self.loss_centerness(pred_centerness, gt_centerness, avg_factor=max(torch.sum(gt_centerness > 0.5), 1))
        
        return centerness_loss
    
    def get_targets(self,
                    gt_centerness,
                    pred_centerness):
        gt_centerness = torch.stack(gt_centerness, dim=0)
        B, N, H, W = gt_centerness.shape
        pred_centerness = F.interpolate(pred_centerness, (H, W), mode='bilinear', align_corners=True)
        gt_centerness = gt_centerness.reshape(B*N, -1, 1)
        
        gt_centerness = gt_centerness.double()
        gt_centerness = torch.where(gt_centerness > 0.9, 1.0, gt_centerness)
        # gt_centerness = gt_centerness.reshape(B*N, -1, 1)
        
        pred_centerness = pred_centerness.permute(0, 2, 3, 1).reshape(B*N, -1, 1) 
        # pred_centerness = pred_centerness.permute(0, 2, 3, 1).reshape(B*N, -1, 1) 
        return gt_centerness, pred_centerness