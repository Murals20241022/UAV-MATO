import torch
import torch.nn as nn
import torch.nn.functional as F

class StemBlock(nn.Module):
    """
    轻量级骨干 Stem Block：统一通道数减少内存消耗，对应论文 §III.B.1
    输出：Stem1→H/2×W/2×16, Stem2→H/2×W/2×32, Stem3→H/4×W/4×64
    """
    def __init__(self, stem_id=1):
        super().__init__()
        self.stem_id = stem_id
        if stem_id == 1:
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)  # H/2
            self.path1 = nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=0)
            self.path2 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=1, stride=1),
                nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
            )
            self.fuse = nn.Conv2d(16, 16, kernel_size=1, stride=1)  # Eltwise融合后通道规整
        elif stem_id == 2:
            self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.path1 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0)
            self.path2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            )
            self.fuse = nn.Conv2d(32, 32, kernel_size=1, stride=1)  # Concat融合后通道规整
        elif stem_id == 3:
            self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            self.path1 = nn.MaxPool2d(kernel_size=2, stride=2)  # H/4
            self.path2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # H/4
            )
            self.fuse = nn.Conv2d(64, 64, kernel_size=1, stride=1)  # Concat(32+32)→64
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.stem_id == 1:
            p1 = self.path1(x)
            p2 = self.path2(x)
            fuse_feat = F.relu(self.fuse(p1 + p2))  # Eltwise融合
        elif self.stem_id == 2:
            p1 = self.path1(x)
            p2 = self.path2(x)
            fuse_feat = F.relu(self.fuse(torch.cat([p1, p2], dim=1)))  # Concat融合
        else:
            p1 = self.path1(x)
            p2 = self.path2(x)
            fuse_feat = F.relu(self.fuse(torch.cat([p1, p2], dim=1)))  # Concat融合
        return fuse_feat

class StageBlock(nn.Module):
    """
    通道堆叠 Stage Block：渐进增加通道数，对应论文 §III.B.2
    输入：H/4×W/4×64，输出：根据堆叠深度调整（论文默认深度=3，输出通道逐步+32）
    """
    def __init__(self, in_channels=64, stack_depth=3, channel_step=32):
        super().__init__()
        self.stack_depth = stack_depth
        self.blocks = nn.ModuleList()
        for d in range(stack_depth):
            out_channels = in_channels + channel_step * (d + 1)
            self.blocks.append(nn.Sequential(
                # 分支1：局部空间特征
                nn.Conv2d(in_channels if d==0 else (in_channels + channel_step*d), 
                          out_channels//3, kernel_size=1, stride=1),
                nn.Conv2d(out_channels//3, out_channels//3, kernel_size=3, stride=1, padding=1),
                # 分支2：边缘细节保留
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                # 分支3：通道优化
                nn.Conv2d(in_channels if d==0 else (in_channels + channel_step*d), 
                          out_channels//3, kernel_size=1, stride=1),
                nn.Conv2d(out_channels//3, out_channels//3, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_channels//3, out_channels//3, kernel_size=1, stride=1),
                # 通道注意力校准
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
            ))
    
    def forward(self, x):
        feat = x
        for block in self.blocks:
            # 多分支特征提取
            branch1 = block[0](feat)
            branch1 = block[1](branch1)
            branch2 = block[2](feat)
            branch3 = block[3](feat)
            branch3 = block[4](branch3)
            branch3 = block[5](branch3)
            # 特征融合与校准
            fuse_feat = torch.cat([branch1, branch2, branch3], dim=1)
            feat = F.relu(block[6](fuse_feat))
        return feat

class LightweightBackbone(nn.Module):
    """整体轻量级骨干：Stem1→Stem2→Stem3→StageBlock×3"""
    def __init__(self, stack_depth=3):
        super().__init__()
        self.stem1 = StemBlock(stem_id=1)
        self.stem2 = StemBlock(stem_id=2)
        self.stem3 = StemBlock(stem_id=3)
        self.stage = StageBlock(in_channels=64, stack_depth=stack_depth)
    
    def forward(self, x):
        x = self.stem1(x)
        x = self.stem2(x)
        x = self.stem3(x)
        x = self.stage(x)
        return x
