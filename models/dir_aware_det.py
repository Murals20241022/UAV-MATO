import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import LightweightBackbone
from .semantic_enhance import SemanticEnhanceModule
from .direction_head import EquivariantRotationConv

class DirAwareDetector(nn.Module):
    """
    整体方向感知检测模型：
    轻量级骨干 → 语义增强 → 等变旋转卷积 → 多任务无锚框头
    """
    def __init__(self, num_classes=3, stack_depth=3, in_channels=3):
        super().__init__()
        self.num_classes = num_classes
        # 1. 轻量级骨干网络
        self.backbone = LightweightBackbone(stack_depth=stack_depth)
        # 2. 语义特征增强模块（骨干输出通道根据stack_depth计算，默认depth=3时为64+32*3=160）
        self.semantic_enhance = SemanticEnhanceModule(in_channels=160)
        # 3. 等变旋转向量场卷积
        self.rot_conv = EquivariantRotationConv(in_channels=160, out_channels=256)
        # 4. 多任务无锚框检测头（分类+回归）
        self.cls_head = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)  # 中心分类
        self.angle_head = nn.Conv2d(256, 1, kernel_size=1, stride=1)          # 角度回归
        self.dir_head = nn.Conv2d(256, 1, kernel_size=1, stride=1)            # 方向回归
        self.size_head = nn.Conv2d(256, 2, kernel_size=1, stride=1)           # 大小回归(w,h)
        self.offset_head = nn.Conv2d(256, 2, kernel_size=1, stride=1)         # 偏移回归(Δx,Δy)
        # 5. 上采样（恢复到输入尺寸的1/4，对应骨干下采样倍数）
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        # 1. 骨干特征提取
        backbone_feat = self.backbone(x)  # (B, 160, H/4, W/4)
        # 2. 语义特征增强
        enhance_feat = self.semantic_enhance(backbone_feat)  # (B, 160, H/4, W/4)
        # 3. 等变旋转特征学习
        rot_feat = self.rot_conv(enhance_feat)  # (B, 256, H/4, W/4)
        # 4. 多任务预测
        cls_pred = self.cls_head(rot_feat)  # (B, C, H/4, W/4)
        angle_pred = self.angle_head(rot_feat)  # (B, 1, H/4, W/4)
        dir_pred = self.dir_head(rot_feat)      # (B, 1, H/4, W/4)
        size_pred = self.size_head(rot_feat)    # (B, 2, H/4, W/4)
        offset_pred = self.offset_head(rot_feat)# (B, 2, H/4, W/4)
        # 5. 上采样到输入尺寸的1/4（后续计算边界框时映射回原图）
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
