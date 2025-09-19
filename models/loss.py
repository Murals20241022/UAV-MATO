import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeightedLoss(nn.Module):
    """
    自适应加权损失：对应论文 §III.A.3 & Eq.(3)-(5)
    包含：中心分类损失(交叉熵) + 4个回归损失(smooth L1)
    """
    def __init__(self, num_classes=3):  # 车辆/卡车/自行车（UAV-MATO）
        super().__init__()
        self.num_classes = num_classes
        # 损失平衡参数δ（初始化：截断正态分布 N(1.0, 0.1)，范围[0.5,1.5]）
        self.delta1 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5))  # 分类
        self.delta2 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5))  # 角度
        self.delta3 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5))  # 方向
        self.delta4 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5))  # 大小
        self.delta5 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5))  # 偏移
    
    def forward(self, preds, targets):
        """
        Args:
            preds: 模型输出 -> dict{
                'cls': (B, H, W, num_classes), 中心分类
                'angle': (B, H, W, 1), 角度回归
                'dir': (B, H, W, 1), 方向回归
                'size': (B, H, W, 2), 大小回归(w,h)
                'offset': (B, H, W, 2) 偏移回归(Δx,Δy)
            }
            targets: 标签 -> 同preds结构
        """
        # 1. 中心分类损失（交叉熵）
        cls_pred = preds['cls'].permute(0, 3, 1, 2)  # (B, C, H, W)
        cls_target = targets['cls'].long()
        cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='mean')
        
        # 2. 回归损失（smooth L1）
        angle_loss = F.smooth_l1_loss(preds['angle'], targets['angle'], reduction='mean')
        dir_loss = F.smooth_l1_loss(preds['dir'], targets['dir'], reduction='mean')
        size_loss = F.smooth_l1_loss(preds['size'], targets['size'], reduction='mean')
        offset_loss = F.smooth_l1_loss(preds['offset'], targets['offset'], reduction='mean')
        
        # 3. 自适应加权总损失（Eq.5）
        total_loss = (1 / (self.delta1 ** 2)) * cls_loss + \
                    (1 / (self.delta2 ** 2)) * angle_loss + \
                    (1 / (self.delta3 ** 2)) * dir_loss + \
                    (1 / (self.delta4 ** 2)) * size_loss + \
                    (1 / (self.delta5 ** 2)) * offset_loss
        return total_loss, {
            'cls_loss': cls_loss.item(),
            'angle_loss': angle_loss.item(),
            'dir_loss': dir_loss.item(),
            'size_loss': size_loss.item(),
            'offset_loss': offset_loss.item()
        }
