import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from .. import builder
from ..builder import SEGMENTORS

# this segmentor might be useful when using it in MMSeg
@SEGMENTORS.register_module()
class PolypSegmentation(BaseModule):
    def __init__(self,
                 backbone,
                 pretrained,
                 decode_head,
                 init_cfg=None,
                 train_cfg=dict(),
                 test_cfg=dict(mode='whole')):
        super().__init__(init_cfg)
        self.backbone = builder.build_backbone(backbone)
        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')
            state_dict = checkpoint
            self.backbone.load_state_dict(state_dict, strict=False)
            loaded_keys = []
            not_loaded_keys = []
            for k in state_dict.keys():
                if k in self.backbone.state_dict():
                    loaded_keys.append(k)
                else:
                    not_loaded_keys.append(k)
        self._init_decode_head(decode_head)


    def _init_decode_head(self, decode_head):
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes


    def forward(self, img):
        out = self.backbone(img)
        out = self.decode_head(out)
        return out