import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction, in_channels, kernel_size=1, stride=1)
        )
    
    def forward(self, x):
        avg_feat = self.avg_pool(x)
        max_feat = self.max_pool(x)
        avg_attn = self.mlp(avg_feat)
        max_attn = self.mlp(max_feat)
        attn = torch.sigmoid(avg_attn + max_attn)
        return x * attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
    
    def forward(self, x):
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        concat_feat = torch.cat([avg_feat, max_feat], dim=1)
        attn = torch.sigmoid(self.conv(concat_feat))
        return x * attn

class ResidualAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 2, 1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 2, 1), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 2, 1), nn.ReLU())
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res_conv = nn.Conv2d(in_channels, in_channels, 1, 1)
    
    def forward(self, x):
        orig = x
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u1 = self.up1(d3) + d2
        u2 = self.up2(u1) + d1
        u3 = self.up3(u2)
        return orig + self.res_conv(u3)

class HierarchicalContext(nn.Module):
    def __init__(self, in_channels, alpha=1, beta=2):
        super().__init__()
        self.alpha = alpha  
        self.beta = beta    
        self.conv = nn.Conv2d(in_channels * 4, in_channels, 1, 1) 
    
    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        up_alpha = F.interpolate(x, size=(H*self.alpha, W*self.alpha), mode='bilinear', align_corners=True)
        up_beta = F.interpolate(x, size=(H*self.beta, W*self.beta), mode='bilinear', align_corners=True)
        down_alpha = F.interpolate(x, size=(H//self.alpha, W//self.alpha), mode='bilinear', align_corners=True)
        down_beta = F.interpolate(x, size=(H//self.beta, W//self.beta), mode='bilinear', align_corners=True)
        up_alpha = F.interpolate(up_alpha, size=(H, W), mode='bilinear', align_corners=True)
        up_beta = F.interpolate(up_beta, size=(H, W), mode='bilinear', align_corners=True)
        down_alpha = F.interpolate(down_alpha, size=(H, W), mode='bilinear', align_corners=True)
        down_beta = F.interpolate(down_beta, size=(H, W), mode='bilinear', align_corners=True)
        concat_feat = torch.cat([x, up_alpha, up_beta, down_alpha, down_beta], dim=1)
        return self.conv(concat_feat)

class SemanticEnhanceModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.ra = ResidualAttention(in_channels)
        self.hc = HierarchicalContext(in_channels)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        x = self.ra(x)
        x = self.hc(x)
        return x
