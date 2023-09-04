# -*- coding: utf-8 -*-
# @Time: 2023/8/3 上午9:28
# @Author: YANG.C
# @File: distillation.py

import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import math
import numpy as np

from ..builder import LOSSES


@LOSSES.register_module
class DistillationLoss(nn.Module):
    def __init__(self, in_feat=1024, num_classes=6, scale=64, margin=0.35, dropout_rate=0.2, base_weight=1.0,
                 distillation_type='soft', distillation_weight=1.0, tau=1.0):
        super(DistillationLoss, self).__init__()

        # base criterion
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        self.base_weight = base_weight

        self.base_fc = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))
        nn.init.kaiming_uniform_(self.base_fc, a=math.sqrt(1))

        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout_rate)

        # distillation criterion
        self.distillation_type = distillation_type
        self.distillation_weight = distillation_weight
        self.distillation_fc = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.tau = tau
        nn.init.kaiming_uniform_(self.distillation_fc, a=math.sqrt(1))

    def forward(self, features, targets=None, teacher_outputs=None):
        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m
        # get cos(theta)
        cos_theta = F.linear(self.dropout(F.normalize(features)), F.normalize(self.base_fc))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        if not self.training:
            pred_class_logits = cos_theta * self._s
            outputs_kd = F.linear(self.dropout(F.normalize(features)), F.normalize(self.distillation_fc))
            outputs_kd = outputs_kd.clamp(-1, 1)  # for numerical stability
            pred_class_distlliation = outputs_kd * self._s
            pred_class_logits = pred_class_logits + pred_class_distlliation
            return pred_class_logits

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        cos_theta_m = cos_theta_m.type_as(target_logit)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        # import pdb; pdb.set_trace()
        cos_theta[mask] = (hard_example * (self.t + hard_example)).type_as(target_logit)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s

        if self.training:
            base_loss = self.criterion(pred_class_logits, targets) * self.base_weight
            distillation_loss = 0
            if self.distillation_type == 'soft':
                T = self.tau
                # taken from https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py#L100
                # with slight modifications
                outputs_kd = F.linear(self.dropout(F.normalize(features)), F.normalize(self.distillation_fc))
                outputs_kd = outputs_kd.clamp(-1, 1)  # for numerical stability
                outputs_kd = outputs_kd * self._s
                distillation_loss = F.kl_div(
                    F.log_softmax(outputs_kd / T, dim=1),
                    F.log_softmax(teacher_outputs / T, dim=1),
                    reduction='sum',
                    log_target=True
                ) * (T * T) / outputs_kd.numel()
            elif self.distillation_type == 'hard':
                outputs_kd = F.linear(self.dropout(F.normalize(features)), F.normalize(self.distillation_fc))
                outputs_kd = outputs_kd.clamp(-1, 1)  # for numerical stability
                outputs_kd = outputs_kd * self._s
                distillation_loss = F.cross_entropy(
                    outputs_kd, teacher_outputs.argmax(dim=1))
            distillation_loss = distillation_loss * self.distillation_weight
            loss = base_loss + distillation_loss
            return loss, [base_loss, distillation_loss]
        else:
            return pred_class_logits


if __name__ == '__main__':
    import random

    random.seed(42)
    torch.random.manual_seed(42)
    dloss = DistillationLoss(in_feat=512, num_classes=6, scale=64, margin=0.35, dropout_rate=0.2, base_weight=1.0,
                             distillation_type='soft', distillation_weight=1.0, tau=1.0)
    inputs = torch.randn(2, 512)
    target = torch.LongTensor([0, 2])
    t_outputs = torch.tensor([[0.0443, 0.0827, -0.0273, -0.0102, 0.0132, -0.0026],
                              [-0.0537, -0.0268, 0.0176, -0.0843, -0.0239, -0.0030]])
    loss = dloss(inputs, target, t_outputs)
    print(loss)
