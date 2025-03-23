
import torch.nn as nn
from .DualAttention import DualAttention

# 改进的可逆块：引入特征解耦
class ReversibleBlock(nn.Module):
    def __init__(self, in_channels, expansion=4, dropout=0.4):
        super().__init__()
        hidden_channels = in_channels * expansion

        # F函数处理通道维度
        self.F = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1),  # 使用1x1卷积代替Linear
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, in_channels, 1)
        )

        # G函数处理空间维度
        self.G = nn.Sequential(
            DualAttention(in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # 添加通道分割后的LayerNorm
        self.norm = nn.LayerNorm(in_channels)  # 针对通道维度做归一化

    def forward(self, x1, x2):
        # x1, x2: (B, C, H, W)
        y1 = x1 + self.F(self.norm(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))  # 正确维度转换
        y2 = x2 + self.G(y1)
        return y1, y2

    def inverse(self, y1, y2):
        x2 = y2 - self.G(y1)
        x1 = y1 - self.F(self.norm(x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x1, x2