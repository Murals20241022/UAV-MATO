import torch
import torch.nn as nn
import torch.nn.functional as F

class EquivariantRotationConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, num_rotations=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations      
        self.base_conv = nn.Conv2d(in_channels, out_channels * num_rotations, 
                                   kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.rot_alpha = nn.Parameter(torch.randn(num_rotations) * 0.1 + 1.0)  
    
    def bilinear_resample(self, filter, angle):
        theta = torch.tensor([[torch.cos(angle), -torch.sin(angle), 0],
                              [torch.sin(angle), torch.cos(angle), 0]], device=filter.device)
        theta = theta.unsqueeze(0).repeat(filter.shape[0], 1, 1)
        grid = F.affine_grid(theta, filter.shape, align_corners=True)
        resampled_filter = F.grid_sample(filter, grid, align_corners=True)
        return resampled_filter
    
    def forward(self, x):
        batch_size = x.shape[0]
        base_feat = self.base_conv(x)  # (B, outC*num_rot, H, W)
        base_feat = base_feat.view(batch_size, self.out_channels, self.num_rotations, x.shape[2], x.shape[3])
        
        rotated_feats = []
        for i in range(self.num_rotations):
            angle = self.rot_alpha[i] * (torch.pi / 4 * i) 
            filter_i = self.base_conv.weight.view(self.out_channels, self.num_rotations, 
                                                  self.in_channels, self.kernel_size, self.kernel_size)[:, i, :, :, :]
            rotated_filter = self.bilinear_resample(filter_i.unsqueeze(2).unsqueeze(3), angle)
            rotated_filter = rotated_filter.squeeze(2).squeeze(3)
            feat_i = F.conv2d(x, rotated_filter, padding=self.kernel_size//2)
            rotated_feats.append(feat_i)
        
        rotated_feats = torch.stack(rotated_feats, dim=2)  # (B, outC, num_rot, H, W)
        max_feat, _ = torch.max(rotated_feats, dim=2)      # (B, outC, H, W)
        return max_feat
