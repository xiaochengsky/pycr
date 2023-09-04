# -*- coding: utf-8 -*-
# @Time : 2022/11/29 下午4:47
# @Author : YANG.C
# @File : cross_entropy.py

import torch
import torch.nn as nn

from ..builder import LOSSES


# 合并 FC + CE
@LOSSES.register_module
class MultiLabelBCE(nn.Module):
    def __init__(self, in_feat=512, num_classes=46, weight=1.0, invalid=45):
        super(MultiLabelBCE, self).__init__()
        self.linear = nn.Linear(in_feat, num_classes, bias=False)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss_weight = []
        for i in range(num_classes):
            if i < 43:
                loss_weight.append(1.0)
            else:
                loss_weight.append(0.1)
        self.weight = torch.tensor(loss_weight)
        self.invalid = invalid

    def forward(self, predicts, targets=None):
        predicts = self.linear(predicts)

        # 训练模式, 输出 CE loss
        if self.training:
            mask = targets != self.invalid
            predicts[~mask] = 1
            targets[~mask] = 1
            loss = self.criterion(predicts, targets.float()) * self.weight
            loss *= mask
            return loss.mean()
        else:
            return predicts

    def simple_test(self, predicts):
        predicts = self.linear(predicts)
        return predicts
