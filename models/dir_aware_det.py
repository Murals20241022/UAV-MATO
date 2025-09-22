import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import LightweightBackbone
from .semantic_enhance import SemanticEnhanceModule
from .direction_head import EquivariantRotationConv

class DirAwareDetector(nn.Module):
    def __init__(self, num_classes=3, stack_depth=3, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = LightweightBackbone(stack_depth=stack_depth)
        self.semantic_enhance = SemanticEnhanceModule(in_channels=160)
        self.rot_conv = EquivariantRotationConv(in_channels=160, out_channels=256)
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)  
        self.angle_head = nn.Conv2d(256, 1, kernel_size=1, stride=1)          
        self.dir_head = nn.Conv2d(256, 1, kernel_size=1, stride=1)        
        self.size_head = nn.Conv2d(256, 2, kernel_size=1, stride=1)          
        self.offset_head = nn.Conv2d(256, 2, kernel_size=1, stride=1)         
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        backbone_feat = self.backbone(x)  # (B, 160, H/4, W/4)
        enhance_feat = self.semantic_enhance(backbone_feat)  # (B, 160, H/4, W/4)
        rot_feat = self.rot_conv(enhance_feat)  # (B, 256, H/4, W/4)
        cls_pred = self.cls_head(rot_feat)  # (B, C, H/4, W/4)
        angle_pred = self.angle_head(rot_feat)  # (B, 1, H/4, W/4)
        dir_pred = self.dir_head(rot_feat)      # (B, 1, H/4, W/4)
        size_pred = self.size_head(rot_feat)    # (B, 2, H/4, W/4)
        offset_pred = self.offset_head(rot_feat)# (B, 2, H/4, W/4)
        cls_pred = self.upsample(cls_pred).permute(0, 2, 3, 1)  # (B, H/4, W/4, C)
        angle_pred = self.upsample(angle_pred).permute(0, 2, 3, 1)
        dir_pred = self.upsample(dir_pred).permute(0, 2, 3, 1)
        size_pred = self.upsample(size_pred).permute(0, 2, 3, 1)
        offset_pred = self.upsample(offset_pred).permute(0, 2, 3, 1)
        return {
            'cls': cls_pred,
            'angle': angle_pred,
            'dir': dir_pred,
            'size': size_pred,
            'offset': offset_pred
        }
