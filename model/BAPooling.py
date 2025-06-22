import torch
import torch.nn as nn
import DualAttention
class BilinearAttentionPoolingModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 双线性分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            DualAttention(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1),
            DualAttention(256),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # 双线性处理层
        self.process = nn.Sequential(
            nn.Linear(256 * 256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        feat1 = self.branch1(x).flatten(1)  # (B, 256)
        feat2 = self.branch2(x).flatten(1)  # (B, 256)

        # 计算双线性交互
        bilinear = torch.bmm(feat1.unsqueeze(2), feat2.unsqueeze(1))  # (B, 256, 256)
        bilinear = bilinear.flatten(1)  # (B, 256*256)
        bilinear = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + 1e-5)

        return self.process(bilinear)  # (B, 512)