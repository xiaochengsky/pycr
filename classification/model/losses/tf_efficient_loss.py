import torch
import torch.nn as nn


# from ..backbone.timm_efficientnets import *


class TF_EfficientNet_SnapMixLoss_LabelSmooth(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, lb_smooth=0.1, reduction='mean', ignore_index=-100, weight=1.0):
        super(TF_EfficientNet_SnapMixLoss_LabelSmooth, self).__init__()

        self.linear = nn.Linear(in_feat, num_classes)
        # self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        self.lb_smooth = lb_smooth
        self.lb_ignore = ignore_index
        self.weight = weight
        self.reduction = reduction
        self.num_classes = num_classes
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, predicts, ya, yb, lam_a, lam_b, reback=None):
        predicts = self.linear(predicts)

        # if self.training:
        #     loss_a = self.criterion(predicts, ya)
        #     loss_b = self.criterion(predicts, yb)
        #     loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        #     return loss * self.weight

        if self.training:
            if reback is None:
                with torch.no_grad():
                    ya = ya.clone().detach()
                    lb_pos_ya, lb_neg_ya = 1. - self.lb_smooth, self.lb_smooth / self.num_classes
                    lb_one_hot_ya = torch.empty_like(predicts).fill_(lb_neg_ya).scatter_(1, ya.unsqueeze(1),
                                                                                   lb_pos_ya).detach()
                    yb = yb.clone().detach()
                    lb_pos_yb, lb_neg_yb = 1. - self.lb_smooth, self.lb_smooth / self.num_classes
                    lb_one_hot_yb = torch.empty_like(predicts).fill_(lb_neg_yb).scatter_(1, yb.unsqueeze(1),
                                                                                        lb_pos_yb).detach()
                logs = self.log_softmax(predicts)
                loss_a = -torch.sum(logs * lb_one_hot_ya, dim=1)
                loss_b = -torch.sum(logs * lb_one_hot_yb, dim=1)
                if self.reduction == 'mean':
                    loss = torch.mean(loss_a * lam_a + loss_b * lam_b)

                else:
                    loss = torch.sum(loss_a * lam_a + loss_b * lam_b)
                return loss * self.weight
            elif reback == 'eval_cls_backbone':
                return predicts

        else:
            return predicts


class TF_EfficientNet_SnapMixLoss(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, weight=1.0):
        super(TF_EfficientNet_SnapMixLoss, self).__init__()

        self.linear = nn.Linear(in_feat, num_classes)
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        self.weight = weight

    # def forward(self, criterion, outputs, ya, yb, lam_a, lam_b):
    #     loss_a = criterion(outputs, ya)
    #     loss_b = criterion(outputs, yb)
    #     loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
    #     return loss

    def forward(self, predicts, ya, yb, lam_a, lam_b, reback=None):
        predicts = self.linear(predicts)

        # if self.training:
        #     loss_a = self.criterion(predicts, ya)
        #     loss_b = self.criterion(predicts, yb)
        #     loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
        #     return loss * self.weight

        if self.training:
            if reback is None:
                loss_a = self.criterion(predicts, ya)
                loss_b = self.criterion(predicts, yb)
                loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
                # return loss * self.weight
                return loss
            elif reback == 'eval_cls_backbone':
                return predicts

        else:
            return predicts


class TF_EfficientNet_CELoss(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, weight=1.0):
        super(TF_EfficientNet_CELoss, self).__init__()

        self.linear = nn.Linear(in_feat, num_classes)
        self.criterion = nn.CrossEntropyLoss(reduction='none').cuda()
        self.weight = weight

    # def forward(self, criterion, outputs, ya, yb, lam_a, lam_b):
    #     loss_a = criterion(outputs, ya)
    #     loss_b = criterion(outputs, yb)
    #     loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
    #     return loss

    def forward(self, predicts, targets=None):
        predicts = self.linear(predicts)

        # 训练模式, 输出 CE loss
        if self.training:
            return self.criterion(predicts, targets) * self.weight

        # 测试模式, 由 FC 输出
        else:
            return predicts
