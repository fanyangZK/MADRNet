
import torch
import torch.nn as nn


# 改进的混合注意力模块：融合通道与空间注意力
class DualAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        # 顶体区域特征编码（输入应为全局压缩后的特征）
        self.acrosome_encoder = nn.Linear(in_dim, in_dim)  # 输入维度改为 in_dim
        self.pool = nn.AdaptiveAvgPool1d(1024)

        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, in_dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(in_dim // 16, in_dim, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # 顶体先验增强（修正输入）
        acrosome_feat = x.mean(dim=[2, 3])  # 全局平均池化，形状 (B, C)
        acrosome_mask = torch.sigmoid(self.acrosome_encoder(acrosome_feat))  # 形状 (B, C)
        acrosome_mask = acrosome_mask.view(B, C, 1, 1)  # 调整为 (B, C, 1, 1)

        # 注意力计算
        channel_scale = self.channel_attn(x) * acrosome_mask
        spatial_scale = self.spatial_attn(x)
        return x * channel_scale + x * spatial_scale