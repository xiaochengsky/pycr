import torch
import torch.nn as nn


class SnapMixLoss(nn.Module):
    def __init__(self, in_feat=2048, num_classes=5, weight=1.0):
        super(SnapMixLoss, self).__init__()
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

