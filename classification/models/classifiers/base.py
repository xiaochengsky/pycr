# -*- coding: utf-8 -*-
# @Time : 2022/11/29 下午8:15
# @Author : YANG.C
# @File : base.py
from abc import ABCMeta, abstractmethod
import copy

import torch
import torch.nn as nn

from typing import Optional


class BaseClassifier(nn.Module):
    def __init__(self, init_cfg: Optional[dict] = None):
        """Initialize BaseClassifier, inherited from `torch.nn.Module`"""
        super().__init__()

        self.is_init = False
        self.init_cfg = copy.deepcopy(init_cfg)

    @abstractmethod
    def extract_feat(self, imgs, stage=None):
        pass

    @abstractmethod
    def forward_train(self, imgs, gt_labels, teacher_outputs, **kwargs):
        pass

    @abstractmethod
    def simple_test(self, imgs, **kwargs):
        pass

    def forward_test(self, imgs, **kwargs):
        """for TTA"""
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]

        if len(imgs) == 1:
            return self.simple_test(imgs[0], **kwargs)
        else:
            raise NotImplementedError('TTA has not been implemented!!!')

    def forward(self, img, gt_labels=None, teacher_outputs=None, return_loss=False, **kwargs):
        if self.training:
            return self.forward_train(img, gt_labels, teacher_outputs, **kwargs)
        else:
            return self.forward_test(img, **kwargs)
