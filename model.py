import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from mix_transformer import MixVisionTransformer
from bfps_head import BFPSHead

class BFPS(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, backbone='b4'):
        super(BFPS, self).__init__()
        
        depths_dict = {
            'b1': [2, 2, 2, 2],
            'b2': [3, 4, 6, 3],
            'b3': [3, 4, 18, 3],
            'b4': [3, 8, 27, 3],
            'b5': [3, 6, 40, 3]
        }
        backbone_depths = depths_dict[backbone]
        self.encoder = MixVisionTransformer(
            in_chans=in_channels,
            patch_size=4,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=backbone_depths,
            sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0,
            drop_path_rate=0.1
        )
        self.encoder.init_weights(f'pretrained/mit_{backbone}.pth')

        self.decoder = BFPSHead(feature_channels=[64, 128, 320, 512], num_classes=num_classes)
    
    
    def forward(self, x):
        features = self.encoder(x)
        out = self.decoder(features)
        return out
