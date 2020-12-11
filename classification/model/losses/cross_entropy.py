import torch
import torch.nn as nn


# 合并 FC + CE
class CrossEntroy(nn.Module):
    def __init__(self, in_feat=512, num_classes=5, weight=1.0):
        super(CrossEntroy, self).__init__()
        self.linear = nn.Linear(in_feat, num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, predicts, targets=None):
        predicts = self.linear(predicts)

        # 训练模式, 输出 CE loss
        if self.training:
            return self.criterion(predicts, targets) * self.weight

        # 测试模式, 由 FC 输出
        else:
            return predicts

