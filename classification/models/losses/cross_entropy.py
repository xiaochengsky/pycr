# -*- coding: utf-8 -*-
# @Time : 2022/11/29 下午4:47
# @Author : YANG.C
# @File : cross_entropy.py

import torch
import torch.nn as nn

from ..builder import LOSSES

from classification.models.losses.arcface import calc_fairness


@LOSSES.register_module
class CrossEntropySplitFC(nn.Module):
    def __init__(self, in_feat=512, hidden=256, dropout_rate=0.15, num_classes1=7, num_classes2=15, weight=[1.0, 1.0]):
        super(CrossEntropySplitFC, self).__init__()
        self.num_classes1 = num_classes1
        self.num_classes2 = num_classes2

        self.hidden_linear1 = nn.Linear(in_feat, hidden)
        self.linear1 = nn.Linear(hidden, num_classes1)

        self.hidden_linear2 = nn.Linear(in_feat, hidden)
        self.linear2 = nn.Linear(hidden, num_classes2)

        self.dropout = nn.Dropout(p=dropout_rate)

        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, predicts, targets=None):
        predicts_classes1 = self.linear1(self.hidden_linear1(self.dropout(predicts)))  # [N, 7]
        predicts_classes2 = self.linear2(self.hidden_linear2(self.dropout(predicts)))  # [N, 15]

        if self.training:
            mask = targets < self.num_classes1
            if mask.sum() == len(mask):
                loss1 = self.criterion(predicts_classes1[mask], targets.long()[mask]) * self.weight[0]
                loss2 = 0
            elif mask.sum() == 0:
                loss1 = 0
                loss2 = self.criterion(predicts_classes2[(~mask)], (targets.long()[~mask]) - self.num_classes1) * \
                        self.weight[1]
            else:
                loss1 = self.criterion(predicts_classes1[mask], targets.long()[mask]) * self.weight[0]
                loss2 = self.criterion(predicts_classes2[(~mask)], (targets.long()[~mask]) - self.num_classes1) * \
                        self.weight[1]
            return loss1 + loss2

        else:
            return [
                predicts_classes1,
                predicts_classes2
            ]


# 合并 FC + CE
@LOSSES.register_module
class CrossEntropy(nn.Module):
    def __init__(self, in_feat=512, num_classes=5, weight=1.0):
        super(CrossEntropy, self).__init__()
        self.linear = nn.Linear(in_feat, num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, predicts, targets=None, groups=None):
        predicts = self.linear(predicts)

        # 训练模式, 输出 CE loss
        if self.training:
            # print(predicts.shape)
            # print(targets.shape, targets)
            # exit(0)
            loss_ce = self.criterion(predicts, targets.long()) * self.weight
            loss_fn = calc_fairness(predicts, targets.long(), groups)
            return loss_ce + loss_fn * 1e10
            # return loss_ce
        else:
            return predicts

    def simple_test(self, predicts):
        predicts = self.linear(predicts)
        return predicts


@LOSSES.register_module
class CrossEntropyDropout(nn.Module):
    def __init__(self, in_feat=512, num_classes=5, dropout_rate=0.15, weight=1.0):
        super(CrossEntropyDropout, self).__init__()
        self.linear = nn.Linear(in_feat, num_classes, bias=False)
        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.weight = weight

    def forward(self, predicts, targets=None):
        predicts = self.linear(self.dropout(predicts))

        # 训练模式, 输出 CE loss
        if self.training:
            # print(predicts.shape)
            # print(targets.shape, targets)
            # exit(0)
            return self.criterion(predicts, targets.long()) * self.weight
        else:
            return predicts

    def simple_test(self, predicts):
        predicts = self.linear(predicts)
        return predicts


@LOSSES.register_module
class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, lb_smooth=0.1, reduction='mean', ignore_index=-100, weight=1.0):
        super(LabelSmoothCrossEntropy, self).__init__()
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


@LOSSES.register_module
class LabelSmoothCrossEntropyDropout(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, dropout_rate=0.15, lb_smooth=0.1, reduction='mean',
                 ignore_index=-100, weight=1.0):
        super(LabelSmoothCrossEntropyDropout, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.num_classes = num_classes
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.linear = nn.Linear(in_feat, num_classes, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, predicts, targets=None):
        predicts = self.linear(self.dropout(predicts))

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
