import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeightedLoss(nn.Module):
    def __init__(self, num_classes=3):  
        super().__init__()
        self.num_classes = num_classes
        self.delta1 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5))  
        self.delta2 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5))  
        self.delta3 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5)) 
        self.delta4 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5)) 
        self.delta5 = nn.Parameter(torch.nn.init.trunc_normal_(torch.empty(1), mean=1.0, std=0.1, a=0.5, b=1.5))  
    
    def forward(self, preds, targets):
        cls_pred = preds['cls'].permute(0, 3, 1, 2)  # (B, C, H, W)
        cls_target = targets['cls'].long()
        cls_loss = F.cross_entropy(cls_pred, cls_target, reduction='mean')
        
        angle_loss = F.smooth_l1_loss(preds['angle'], targets['angle'], reduction='mean')
        dir_loss = F.smooth_l1_loss(preds['dir'], targets['dir'], reduction='mean')
        size_loss = F.smooth_l1_loss(preds['size'], targets['size'], reduction='mean')
        offset_loss = F.smooth_l1_loss(preds['offset'], targets['offset'], reduction='mean')
        
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
