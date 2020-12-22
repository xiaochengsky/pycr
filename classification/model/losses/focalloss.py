import torch
import torch.nn as nn
import torch.nn.functional as F


# 合并 FC + FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, size_average=True, ignore_index=255, in_feat=512, num_classes=5,
                 weight=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.weight = weight
        self.ignore_index = ignore_index
        self.linear = nn.Linear(in_feat, num_classes, bias=False)

    def forward(self, predicts, targets=None):
        predicts = self.linear(predicts)

        # 训练模式, 输出 FocalLoss
        if self.training:
            ce_loss = F.cross_entropy(predicts, targets, reduction="none", ignore_index=self.ignore_index)
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            if self.size_average:
                return focal_loss.mean()
            else:
                return focal_loss.sum()

        # 测试模式, 由 FC 输出
        else:
            return predicts
