import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SnapMixEQLoss(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, weight=1.0, freq_info=[], gamma=None, _lambda=None, device=0):
        super(SnapMixEQLoss, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_feat, num_classes, bias=True)
        self.weight = weight
        if not torch.is_tensor(freq_info):
            self._freq_info = torch.tensor(freq_info, dtype=torch.float)
        self._lambda = _lambda
        self.gamma = gamma
        self.num_inputs = 16

    def exclude_func(self):
        weight_beta = torch.zeros(self.num_classes).cuda(device)
        beta = torch.zeros_like(weight_beta).uniform_()
        weight_beta[beta < self.gamma] = 1
        weight_beta = weight_beta.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        return weight_beta

    def threshold_func(self):
        weight_thre = torch.zeros(self.num_classes, dtype=torch.float).cuda(device)
        weight_thre[self._freq_info < self._lambda] = 1
        weight_thre = weight_thre.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        return weight_thre

    def expand_label(self, predicts, target):
        one_hot = torch.zeros_like(predicts, dtype=torch.float).cuda(device)
        targets = target.unsqueeze(1)
        one_hot = one_hot.scatter_(1, targets, 1)
        return one_hot

    def forward(self, predicts, ya, yb, lam_a, lam_b, reback=None):
        predicts = self.linear(predicts)
        if self.training:
            if reback is None:
                self.num_inputs = predicts.shape[0]
                weight_beta = self.exclude_func()
                weight_thre = self.threshold_func()

                one_hot_target_ya = self.expand_label(predicts, ya)
                one_hot_target_yb = self.expand_label(predicts, yb)
                eql_weight_ya = 1 - weight_beta * weight_thre * (1 - one_hot_target_ya)
                eql_weight_yb = 1 - weight_beta * weight_thre * (1 - one_hot_target_yb)

                e_inputs = torch.exp(predicts)

                sum_exp_ya = torch.sum(eql_weight_ya * e_inputs, dim=1)
                sum_exp_yb = torch.sum(eql_weight_yb * e_inputs, dim=1)
                sum_exp_ya = sum_exp_ya.unsqueeze(1).repeat(1, self.num_classes)
                sum_exp_yb = sum_exp_yb.unsqueeze(1).repeat(1, self.num_classes)
                eq_loss_ya = torch.log(e_inputs / sum_exp_ya)
                eq_loss_yb = torch.log(e_inputs / sum_exp_yb)

                loss = F.nll_loss(eq_loss_ya, ya) * lam_a + F.nll_loss(eq_loss_yb, yb) * lam_b
                loss = torch.mean(loss)
                return loss * self.weight

            elif reback == "eval_cls_backbone":
                return predicts

        else:
            return predicts


class SnapMixEQLossV1(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, weight=1.0, freq_info=[], gamma=None, _lambda=None):
        super(SnapMixEQLossV1, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_feat, num_classes, bias=True)
        self.weight = weight
        if not torch.is_tensor(freq_info):
            self._freq_info = torch.tensor(freq_info, dtype=torch.float)
        self._lambda = _lambda
        self.gamma = gamma
        self.num_inputs = 16

    def exclude_func(self):
        weight_beta = torch.zeros(self.num_classes).cuda()
        beta = torch.zeros_like(weight_beta).uniform_()
        weight_beta[beta < self.gamma] = 1
        weight_beta = weight_beta.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        return weight_beta

    def threshold_func(self):
        weight_thre = torch.zeros(self.num_classes, dtype=torch.float).cuda()
        weight_thre[self._freq_info < self._lambda] = 1
        weight_thre = weight_thre.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        return weight_thre

    def customize_func(self, targets):
        weight_cust = torch.zeros(self.num_classes, dtype=torch.float).uniform_(0, 1).cuda()
        weight_cust = weight_cust.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        weight_cust = torch.bernoulli(weight_cust)
        idx = torch.arange(self.num_inputs)

        # 高类别一定要计算 loss
        weight_cust[idx, 3] = 0

        # 当前类别一定要计算 loss
        weight_cust[idx, targets] = 0
        return weight_cust

    def expand_label(self, predicts, target):
        one_hot = torch.zeros_like(predicts, dtype=torch.float).cuda()
        targets = target.unsqueeze(1)
        one_hot = one_hot.scatter_(1, targets, 1)
        return one_hot

    def forward(self, predicts, ya, yb, lam_a, lam_b, reback=None):
        predicts = self.linear(predicts)
        if self.training:
            if reback is None:
                self.num_inputs = predicts.shape[0]
                # weight_beta = self.exclude_func()
                # weight_thre = self.threshold_func()

                one_hot_target_ya = self.expand_label(predicts, ya)
                one_hot_target_yb = self.expand_label(predicts, yb)

                weight_cust_ya = self.customize_func(ya)
                weight_cust_yb = self.customize_func(yb)
                eql_weight_ya = 1 - weight_cust_ya * (1 - one_hot_target_ya)
                eql_weight_yb = 1 - weight_cust_yb * (1 - one_hot_target_yb)
                e_inputs = torch.exp(predicts)

                sum_exp_ya = torch.sum(eql_weight_ya * e_inputs, dim=1)
                sum_exp_yb = torch.sum(eql_weight_yb * e_inputs, dim=1)
                sum_exp_ya = sum_exp_ya.unsqueeze(1).repeat(1, self.num_classes)
                sum_exp_yb = sum_exp_yb.unsqueeze(1).repeat(1, self.num_classes)
                eq_loss_ya = torch.log(e_inputs / sum_exp_ya)
                eq_loss_yb = torch.log(e_inputs / sum_exp_yb)

                loss = F.nll_loss(eq_loss_ya, ya) * lam_a + F.nll_loss(eq_loss_yb, yb) * lam_b
                loss = torch.mean(loss)
                return loss * self.weight

            elif reback == "eval_cls_backbone":
                return predicts

        else:
            return predicts


class SnapMixEQLossShareWeight(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, weight=1.0, freq_info=[], gamma=None, _lambda=None):
        super(SnapMixEQLossShareWeight, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_feat, num_classes, bias=True)
        self.weight = weight
        if not torch.is_tensor(freq_info):
            self._freq_info = torch.tensor(freq_info, dtype=torch.float)
        self._lambda = _lambda
        self.gamma = gamma
        self.num_inputs = 16

    def exclude_func(self):
        weight_beta = torch.zeros(self.num_classes).cuda()
        beta = torch.zeros_like(weight_beta).uniform_()
        weight_beta[beta < self.gamma] = 1
        weight_beta = weight_beta.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        return weight_beta

    def threshold_func(self):
        weight_thre = torch.zeros(self.num_classes, dtype=torch.float).cuda()
        weight_thre[self._freq_info < self._lambda] = 1
        weight_thre = weight_thre.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        return weight_thre

    def customize_func(self, targets):
        weight_cust = torch.zeros(self.num_classes, dtype=torch.float).uniform_(0, 1).cuda()
        weight_cust = weight_cust.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        weight_cust = torch.bernoulli(weight_cust)
        idx = torch.arange(self.num_inputs)

        # 高类别一定要计算 loss
        weight_cust[idx, 3] = 0

        # 当前类别一定要计算 loss
        weight_cust[idx, targets] = 0
        return weight_cust

    def customize_func_share(self, target_a, target_b):
        weight_cust = torch.zeros(self.num_classes, dtype=torch.float).uniform_(0, 1).cuda()
        weight_cust = weight_cust.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        weight_cust = torch.bernoulli(weight_cust)
        idx = torch.arange(self.num_inputs)

        # 高类别一定要计算 loss
        weight_cust[idx, 3] = 0

        # 当前类别一定要计算 loss
        weight_cust[idx, target_a] = 0
        weight_cust[idx, target_b] = 0
        return weight_cust

    def customize_func_iid(self, target_a, target_b):
        weight_cust = torch.zeros(self.num_classes, dtype=torch.float).uniform_(0, 1).cuda()
        weight_cust = weight_cust.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        weight_cust = torch.bernoulli(weight_cust)
        idx = torch.arange(self.num_inputs)

        # 高类别一定要计算 loss
        weight_cust[idx, 3] = 0

        # 当前类别一定要计算 loss
        weight_cust[idx, target_a] = 0
        # 同一张图的不同类别, 不计算, 避免互斥
        weight_cust[idx, target_b] = 1
        return weight_cust

    def expand_label(self, predicts, target):
        one_hot = torch.zeros_like(predicts, dtype=torch.float).cuda()
        targets = target.unsqueeze(1)
        one_hot = one_hot.scatter_(1, targets, 1)
        return one_hot

    def forward(self, predicts, ya, yb, lam_a, lam_b, reback=None):
        predicts = self.linear(predicts)
        if self.training:
            if reback is None:
                self.num_inputs = predicts.shape[0]
                # weight_beta = self.exclude_func()
                # weight_thre = self.threshold_func()

                one_hot_target_ya = self.expand_label(predicts, ya)
                one_hot_target_yb = self.expand_label(predicts, yb)

                weight_cust = self.customize_func_share(ya, yb)
                eql_weight_ya = 1 - weight_cust * (1 - one_hot_target_ya)
                eql_weight_yb = 1 - weight_cust * (1 - one_hot_target_yb)
                e_inputs = torch.exp(predicts)

                sum_exp_ya = torch.sum(eql_weight_ya * e_inputs, dim=1)
                sum_exp_yb = torch.sum(eql_weight_yb * e_inputs, dim=1)
                sum_exp_ya = sum_exp_ya.unsqueeze(1).repeat(1, self.num_classes)
                sum_exp_yb = sum_exp_yb.unsqueeze(1).repeat(1, self.num_classes)
                eq_loss_ya = torch.log(e_inputs / sum_exp_ya)
                eq_loss_yb = torch.log(e_inputs / sum_exp_yb)

                loss = F.nll_loss(eq_loss_ya, ya) * lam_a + F.nll_loss(eq_loss_yb, yb) * lam_b
                loss = torch.mean(loss)
                return loss * self.weight

            elif reback == "eval_cls_backbone":
                return predicts

        else:
            return predicts


# 添加 iid 条件
class SnapMixEQLossIIDWeight(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, weight=1.0, freq_info=[], gamma=None, _lambda=None, device=0):
        super(SnapMixEQLossIIDWeight, self).__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(in_feat, num_classes, bias=True)
        self.weight = weight
        if not torch.is_tensor(freq_info):
            self._freq_info = torch.tensor(freq_info, dtype=torch.float)
        self._lambda = _lambda
        self.gamma = gamma
        self.num_inputs = 16
        self.device = device

    def exclude_func(self):
        weight_beta = torch.zeros(self.num_classes).cuda(self.device)
        beta = torch.zeros_like(weight_beta).uniform_()
        weight_beta[beta < self.gamma] = 1
        weight_beta = weight_beta.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        return weight_beta

    def threshold_func(self):
        weight_thre = torch.zeros(self.num_classes, dtype=torch.float).cuda(self.device)
        weight_thre[self._freq_info < self._lambda] = 1
        weight_thre = weight_thre.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        return weight_thre

    def customize_func(self, targets):
        weight_cust = torch.zeros(self.num_classes, dtype=torch.float).uniform_(0, 1).cuda(self.device)
        weight_cust = weight_cust.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        weight_cust = torch.bernoulli(weight_cust)
        idx = torch.arange(self.num_inputs)

        # 高类别一定要计算 loss
        weight_cust[idx, 3] = 0

        # 当前类别一定要计算 loss
        weight_cust[idx, targets] = 0
        return weight_cust

    def customize_func_share(self, target_a, target_b):
        weight_cust = torch.zeros(self.num_classes, dtype=torch.float).uniform_(0, 1).cuda(self.device)
        weight_cust = weight_cust.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        weight_cust = torch.bernoulli(weight_cust)
        idx = torch.arange(self.num_inputs)

        # 高类别一定要计算 loss
        weight_cust[idx, 3] = 0

        # 当前类别一定要计算 loss
        weight_cust[idx, target_a] = 0
        weight_cust[idx, target_b] = 0
        return weight_cust

    def customize_func_iid(self, target_a, target_b):
        weight_cust = torch.zeros(self.num_classes, dtype=torch.float).uniform_(0, 1).cuda(self.device)
        weight_cust = weight_cust.view(1, self.num_classes).expand(self.num_inputs, self.num_classes)
        weight_cust = torch.bernoulli(weight_cust)
        idx = torch.arange(self.num_inputs)

        # 高类别一定要计算 loss
        weight_cust[idx, 3] = 0

        # 当前类别一定要计算 loss
        weight_cust[idx, target_a] = 0
        # 同一张图的不同类别, 不计算, 避免互斥
        weight_cust[idx, target_b] = 1
        return weight_cust

    def expand_label(self, predicts, target):
        one_hot = torch.zeros_like(predicts, dtype=torch.float).cuda(self.device)
        targets = target.unsqueeze(1)
        one_hot = one_hot.scatter_(1, targets, 1)
        return one_hot

    def forward(self, predicts, ya, yb, lam_a, lam_b, reback=None):
        predicts = self.linear(predicts)
        if self.training:
            if reback is None:
                self.num_inputs = predicts.shape[0]
                # weight_beta = self.exclude_func()
                # weight_thre = self.threshold_func()

                one_hot_target_ya = self.expand_label(predicts, ya)
                one_hot_target_yb = self.expand_label(predicts, yb)

                weight_cust_ya = self.customize_func_iid(ya, yb)
                weight_cust_yb = self.customize_func_iid(yb, ya)
                eql_weight_ya = 1 - weight_cust_ya * (1 - one_hot_target_ya)
                eql_weight_yb = 1 - weight_cust_yb * (1 - one_hot_target_yb)
                e_inputs = torch.exp(predicts)

                sum_exp_ya = torch.sum(eql_weight_ya * e_inputs, dim=1)
                sum_exp_yb = torch.sum(eql_weight_yb * e_inputs, dim=1)
                sum_exp_ya = sum_exp_ya.unsqueeze(1).repeat(1, self.num_classes)
                sum_exp_yb = sum_exp_yb.unsqueeze(1).repeat(1, self.num_classes)
                eq_loss_ya = torch.log(e_inputs / sum_exp_ya)
                eq_loss_yb = torch.log(e_inputs / sum_exp_yb)

                loss = F.nll_loss(eq_loss_ya, ya) * lam_a + F.nll_loss(eq_loss_yb, yb) * lam_b
                loss = torch.mean(loss)
                return loss * self.weight

            elif reback == "eval_cls_backbone":
                return predicts

        else:
            return predicts

