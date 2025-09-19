import torch
import torch.nn as nn
import torch.nn.functional as F

class EquivariantRotationConv(nn.Module):
    """
    等变旋转向量场卷积：学习 aerial 目标的旋转不变特征
    对应论文 §III.A.1 & Eq.(1)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, num_rotations=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations  # 旋转角度数量(0°/45°/.../315°)
        
        # 基础卷积核（将通过旋转生成多方向核）
        self.base_conv = nn.Conv2d(in_channels, out_channels * num_rotations, 
                                   kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        # 动态角度调整超参数（可学习）
        self.rot_alpha = nn.Parameter(torch.randn(num_rotations) * 0.1 + 1.0)  # 旋转因子
    
    def bilinear_resample(self, filter, angle):
        """双线性重采样：根据角度旋转卷积核，对应论文 §III.A.1 旋转因子处理"""
        theta = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                              [torch.sin(angle), torch.cos(angle), 0]], device=filter.device)
        theta = theta.unsqueeze(0).repeat(filter.shape[0], 1, 1)
        grid = F.affine_grid(theta, filter.shape, align_corners=True)
        resampled_filter = F.grid_sample(filter, grid, align_corners=True)
        return resampled_filter
    
    def forward(self, x):
        # 1. 生成基础卷积特征
        batch_size = x.shape[0]
        base_feat = self.base_conv(x)  # (B, outC*num_rot, H, W)
        base_feat = base_feat.view(batch_size, self.out_channels, self.num_rotations, x.shape[2], x.shape[3])
        
        # 2. 对每个旋转方向的核进行重采样
        rotated_feats = []
        for i in range(self.num_rotations):
            angle = self.rot_alpha[i] * (torch.pi / 4 * i)  # 角度计算（45°间隔）
            # 提取当前旋转方向的卷积核
            filter_i = self.base_conv.weight.view(self.out_channels, self.num_rotations, 
                                                  self.in_channels, self.kernel_size, self.kernel_size)[:, i, :, :, :]
            # 双线性重采样旋转核
            rotated_filter = self.bilinear_resample(filter_i.unsqueeze(2).unsqueeze(3), angle)
            rotated_filter = rotated_filter.squeeze(2).squeeze(3)
            # 应用旋转核计算特征
            feat_i = F.conv2d(x, rotated_filter, padding=self.kernel_size//2)
            rotated_feats.append(feat_i)
        
        # 3. 选择最高激活值的旋转特征（论文 §III.A.1 激活选择策略）
        rotated_feats = torch.stack(rotated_feats, dim=2)  # (B, outC, num_rot, H, W)
        max_feat, _ = torch.max(rotated_feats, dim=2)      # (B, outC, H, W)
        return max_feat
