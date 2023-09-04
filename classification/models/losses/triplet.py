# -*- coding: utf-8 -*-
# @Time: 2023/7/10 下午7:49
# @Author: YANG.C
# @File: triplet.py

import torch
import torch.nn as nn

from ..builder import LOSSES


@LOSSES.register_module
class TripletLoss(nn.Module):
    def __init__(self, margin=0.6, weight=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.weight = weight
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, features, targets=None):
        features = features[:, :, 0, 0]
        n = features.size(0)
        dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
        features = features.to(torch.float32)
        dist = dist + dist.t()
        dist.addmm_(1, -2, features, features.t())  # (a - b)^2
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            if len(dist[i][mask[i]]) > 0 and len(dist[i][mask[i] == 0]) > 0:
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.training:
            return loss * self.weight
        else:
            return None
