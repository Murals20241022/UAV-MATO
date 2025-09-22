import torch
import torch.nn as nn
import torch.nn.functional as F

class StemBlock(nn.Module):
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
            self.fuse = nn.Conv2d(16, 16, kernel_size=1, stride=1)  
        elif stem_id == 2:
            self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
            self.path1 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0)
            self.path2 = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=1, stride=1),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
            )
            self.fuse = nn.Conv2d(32, 32, kernel_size=1, stride=1)  
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
            fuse_feat = F.relu(self.fuse(p1 + p2))  
        elif self.stem_id == 2:
            p1 = self.path1(x)
            p2 = self.path2(x)
            fuse_feat = F.relu(self.fuse(torch.cat([p1, p2], dim=1)))  
        else:
            p1 = self.path1(x)
            p2 = self.path2(x)
            fuse_feat = F.relu(self.fuse(torch.cat([p1, p2], dim=1)))  
        return fuse_feat

class StageBlock(nn.Module):
    def __init__(self, in_channels=64, stack_depth=3, channel_step=32):
        super().__init__()
        self.stack_depth = stack_depth
        self.blocks = nn.ModuleList()
        for d in range(stack_depth):
            out_channels = in_channels + channel_step * (d + 1)
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_channels if d==0 else (in_channels + channel_step*d), 
                          out_channels//3, kernel_size=1, stride=1),
                nn.Conv2d(out_channels//3, out_channels//3, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
                nn.Conv2d(in_channels if d==0 else (in_channels + channel_step*d), 
                          out_channels//3, kernel_size=1, stride=1),
                nn.Conv2d(out_channels//3, out_channels//3, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_channels//3, out_channels//3, kernel_size=1, stride=1),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1)
            ))
    
    def forward(self, x):
        feat = x
        for block in self.blocks:
            branch1 = block[0](feat)
            branch1 = block[1](branch1)
            branch2 = block[2](feat)
            branch3 = block[3](feat)
            branch3 = block[4](branch3)
            branch3 = block[5](branch3)
            fuse_feat = torch.cat([branch1, branch2, branch3], dim=1)
            feat = F.relu(block[6](fuse_feat))
        return feat

class LightweightBackbone(nn.Module):
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
