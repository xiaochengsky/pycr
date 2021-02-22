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



class LabelSmoothCrossEntroy(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, lb_smooth=0.1, reduction='mean', ignore_index=-100, weight=1.0):
        super(LabelSmoothCrossEntroy, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.num_classes = num_classes
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.linear = nn.Linear(in_feat, num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, predicts, targets=None):
        predicts = self.linear(predicts)

        # 训练模式, 输出 CE loss
        if self.training:
            predicts = predicts.float()
            with torch.no_grad():
                targets = targets.clone().detach()
                lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / self.num_classes
                lb_one_hot = torch.empty_like(predicts).fill_(lb_neg).scatter_(1, targets.unsqueeze(1), lb_pos).detach()

            logs = self.log_softmax(predicts)
            loss = -torch.sum(logs * lb_one_hot, dim=1)
            if self.reduction == 'mean':
                loss = loss.mean()
            if self.reduction == 'sum':
                loss = loss.sum()
            return loss * self.weight

        # 测试模式, 由 FC 输出
        else:
            return predicts


