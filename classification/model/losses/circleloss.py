import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class CircleLoss(nn.Module):
    def __init__(self, in_feat, num_classes, gamma=128, m=0.25, weights=1.0):
        super(CircleLoss, self).__init__()
        self.in_feat = in_feat
        self.num_classes = num_classes
        self.gamma = gamma
        self.weights = weights
        self.m = m

        self.weight = Parameter(torch.Tensor(num_classes, in_feat))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, predicts, targets=None):
        sim_mat = F.linear(F.normalize(predicts), F.normalize(self.weight))
        # alpha_p = [Op - Sp]+
        # alpha_n = [Sn - On]+
        # 最优值 Op, On 是根据 m 优化的, 带截断:
        # Op = 1 + m, On = -m
        # 其中 ∆p 和 ∆n 分别为类间和类内的相似度阈值(margin):
        # ∆p = 1 - m，∆n = m
        alpha_p = torch.clamp_min(-sim_mat.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(sim_mat.detach() + self.m, min=0.)
        delta_p = 1 - self.m
        delta_n = self.m

        s_p = self.gamma * alpha_p * (sim_mat - delta_p)
        s_n = self.gamma * alpha_n * (sim_mat - delta_n)

        targets = F.one_hot(targets, num_classes=self._num_classes)

        pred_class_logits = targets * s_p + (1.0 - targets) * s_n

        if self.training:
            all_embedding = pred_class_logits
            all_targets = targets
            dist_mat = torch.matmul(all_embedding, all_embedding.t())

            N = dist_mat.size(0)
            is_pos = all_targets.view(N, 1).expand(N, N).eq(all_targets.view(N, 1).expand(N, N).t()).float()

            # Compute the mask which ignores the relevance score of the query to itself
            is_pos = is_pos - torch.eye(N, N, device=is_pos.device)

            is_neg = all_targets.view(N, 1).expand(N, N).ne(all_targets.view(N, 1).expand(N, N).t())

            s_p = dist_mat * is_pos
            s_n = dist_mat * is_neg

            alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
            alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)
            delta_p = 1 - self.m
            delta_n = self.m

            logit_p = - self.gamma * alpha_p * (s_p - delta_p)
            logit_n = self.gamma * alpha_n * (s_n - delta_n)

            loss = nn.functional.softplus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
            return loss

        else:
            return pred_class_logits


