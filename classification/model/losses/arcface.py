import torch.nn as nn
import math
import torch
import torch.nn.functional as F
import math
import numpy as np


class ArcfaceLoss_Dropout(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, scale=64, margin=0.35, dropout_rate=0.2, weight=1.0):
        super(ArcfaceLoss_Dropout, self).__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        self.weight_loss = weight

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))

        self.criterion = nn.CrossEntropyLoss()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, features, targets):
        # print(self._m)
        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m
        # get cos(theta)
        cos_theta = F.linear(self.dropout(F.normalize(features)), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

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

        # print(pred_class_logits.shape,targets.shape)
        if self.training:
            loss = self.criterion(pred_class_logits, targets) * self.weight_loss
            return loss
        else:
            return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )


class ArcfaceLoss(nn.Module):
    def __init__(self, in_feat, num_classes, scale=64, margin=0.35, weight=1.0):
        super(ArcfaceLoss, self).__init__()
        self.in_feat = in_feat
        self._num_classes = num_classes
        self._s = scale
        self._m = margin
        self.weight_loss = weight

        self.cos_m = math.cos(self._m)
        self.sin_m = math.sin(self._m)
        self.threshold = math.cos(math.pi - self._m)
        self.mm = math.sin(math.pi - self._m) * self._m

        self.weight = nn.Parameter(torch.Tensor(num_classes, in_feat))
        self.register_buffer('t', torch.zeros(1))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, targets):
        # get cos(theta)
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        target_logit = cos_theta[torch.arange(0, features.size(0)), targets].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, targets.view(-1, 1).long(), final_target_logit)
        pred_class_logits = cos_theta * self._s

        # print(pred_class_logits.shape,targets.shape)
        if self.training:
            loss = self.criterion(pred_class_logits, targets) * self.weight_loss
            return loss
        else:
            return pred_class_logits

    def extra_repr(self):
        return 'in_features={}, num_classes={}, scale={}, margin={}'.format(
            self.in_feat, self._num_classes, self._s, self._m
        )


class ArcFaceLoss(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_feat=2048, num_classes=5, s=30.0, m=0.50, easy_margin=False, weight=1.0):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_feat
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feat))
        self.weight_loss = weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, targets, infer=False):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # cosine = cosine.float()

        if infer:
            one_hot = torch.zeros(cosine.size(), device='cuda')
            predicts = (1.0 - one_hot) * cosine
            predicts *= self.s
        else:
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, targets.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            predicts = (one_hot * phi) + (
                        (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            predicts *= self.s
            # print(output)

        if self.training:
            loss = self.criterion(predicts, targets) * self.weight_loss
            return loss
        else:
            return predicts


class ArcFaceSnapMixLoss(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_feat=2048, num_classes=5, s=30.0, m=0.50, easy_margin=False, weight=1.0, device=0):
        super(ArcFaceSnapMixLoss, self).__init__()
        self.in_features = in_feat
        self.out_features = num_classes
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_feat))
        self.bias = nn.Parameter(torch.Tensor(num_classes))
        self.linear
        self.weight_loss = weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(1))
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(1))

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, input, ya, yb, lam_a, lam_b, reback=None, infer=False):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # cosine = self.linear(F.normalize(input), F.normalize(self.linear.weight))
        # cosine = cosine.float()

        if infer:
            one_hot = torch.zeros(cosine.size()).cuda(self.device)
            predicts = (1.0 - one_hot) * cosine
            predicts *= self.s
        else:
            if self.training:
                if reback is None:
                    sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
                    phi = cosine * self.cos_m - sine * self.sin_m
                    if self.easy_margin:
                        phi = torch.where(cosine > 0, phi, cosine)
                    else:
                        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
                    # --------------------------- convert label to one-hot ---------------------------
                    # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
                    one_hot_ya = torch.zeros(cosine.size()).cuda(self.device)
                    one_hot_ya.scatter_(1, ya.view(-1, 1).long(), 1)
                    one_hot_yb = torch.zeros(cosine.size(), device='cuda')
                    one_hot_yb.scatter_(1, yb.view(-1, 1).long(), 1)
                    # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
                    predicts_ya = (one_hot_ya * phi) + (
                                (1.0 - one_hot_ya) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
                    predicts_ya *= self.s
                    predicts_yb = (one_hot_yb * phi) + (
                                (1.0 - one_hot_yb) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
                    predicts_yb *= self.s

                    loss_a = self.criterion(predicts_ya, ya)
                    loss_b = self.criterion(predicts_yb, yb)
                    loss = torch.mean(loss_a * lam_a + loss_b * lam_b)
                    return loss * self.weight_loss

                elif reback == 'eval_cls_backbone':
                    one_hot = torch.zeros(cosine.size()).cuda(self.device)
                    predicts = (1.0 - one_hot) * cosine
                    predicts *= self.s
                    return predicts
            else:
                one_hot = torch.zeros(cosine.size()).cuda(self.device)
                predicts = (1.0 - one_hot) * cosine
                predicts *= self.s
                return predicts


if __name__ == '__main__':
    loss = ArcFaceLoss(2048, 5)
    inputs = torch.randn(2, 2048)
    target = torch.LongTensor([0, 2])
    print(loss(inputs, target))
