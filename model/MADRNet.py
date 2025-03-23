import torch
import torch.nn as nn
from torchvision import  models
from .DualAttention import DualAttention
from .ReversibleBlock import ReversibleBlock
# 核心网络结构
class MADRNet(nn.Module):
    def __init__(self, num_classes=200, backbone='resnet50', img_size=448):
        super().__init__()
        # Backbone初始化
        self.backbone = getattr(models, backbone)(pretrained=True)
        in_channels = 2048  # ResNet50最后一层的输出通道数

        # 修改可逆块输入参数
        self.rev_blocks = nn.ModuleList([
            ReversibleBlock(in_channels // 2) for _ in range(3)  # 确保通道分割正确
        ])

        # 调整双线性池化输入维度
        self.bilinear_pool = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            DualAttention(in_channels // 2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.gap = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512)
        )

        # 修改分类头
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # Backbone特征提取
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)  # 输出形状: (B, 2048, 14, 14)

        # 通道分割
        x1, x2 = torch.chunk(x, 2, dim=1)  # 分割后各为(B, 1024, 14, 14)

        ## 可逆特征融合
        for block in self.rev_blocks:
            x1, x2 = block(x1, x2)

        # 合并特征
        fused_feat = torch.cat([x1, x2], dim=1)  # (B, 2048, 14, 14)

        # 双线性池化
        bilinear_feat = self.bilinear_pool(fused_feat)
        bilinear_feat = bilinear_feat.view(bilinear_feat.size(0), -1)  # (B, 1024)

        # 输出嵌入向量和分类结果
        embeddings = self.embedding_layer(bilinear_feat)
        return self.classifier(embeddings), embeddings