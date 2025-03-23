import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, margin=1.0, ratio_margin=0.3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.7))  # 可学习参数
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.ce = nn.CrossEntropyLoss()
        self.triplet = nn.TripletMarginLoss(margin=margin)
        self.ratio_margin = ratio_margin

    def forward(self, pred, target, embeddings, ratios):
        ce_loss = self.ce(pred, target)

        # 更鲁棒的三元组采样
        unique_classes = torch.unique(target)
        anchors, positives, negatives = [], [], []

        for cls in unique_classes:
            # 当前类别样本作为anchor
            cls_mask = (target == cls)
            if torch.sum(cls_mask) < 2:  # 至少需要两个样本才能构成正样本对
                continue

            # 随机选择anchor和positive
            cls_indices = torch.where(cls_mask)[0]
            anchor_idx, pos_idx = torch.randperm(len(cls_indices))[:2]
            anchor = embeddings[cls_indices[anchor_idx]]
            positive = embeddings[cls_indices[pos_idx]]

            # 随机选择负样本
            neg_cls = unique_classes[unique_classes != cls]
            if len(neg_cls) == 0:
                continue
            neg_cls = neg_cls[torch.randint(0, len(neg_cls), (1,))]
            neg_mask = (target == neg_cls)
            if torch.sum(neg_mask) > 0:
                neg_indices = torch.where(neg_mask)[0]
                negative = embeddings[neg_indices[0]]  # 取第一个负样本

                anchors.append(anchor)
                positives.append(positive)
                negatives.append(negative)

        # 处理有效三元组
        if len(anchors) > 0:
            anchors = torch.stack(anchors)
            positives = torch.stack(positives)
            negatives = torch.stack(negatives)

            # 确保维度一致 (B, D)
            assert anchors.dim() == 2, f"Anchor dim error: {anchors.shape}"
            assert positives.dim() == 2, f"Positive dim error: {positives.shape}"
            assert negatives.dim() == 2, f"Negative dim error: {negatives.shape}"

            triplet_loss = self.triplet(anchors, positives, negatives)
        else:
            triplet_loss = 0.0

        ratio_loss = F.relu(torch.abs(ratios - 1.5) - self.ratio_margin).mean()
        # return ce_loss + 0.5 * triplet_loss + 0.3 * ratio_loss
        return ce_loss + self.alpha * triplet_loss + self.beta * ratio_loss