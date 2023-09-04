# -*- coding: utf-8 -*-
# @Time : 2022/12/6 上午9:48
# @Author : YANG.C
# @File : lr_scheduler.py

import copy

import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import *


class warpper_lr_scheduler(object):
    def __init__(self, cfg_lr_scheduler, optimizer, scheduler_cfg):
        cfg_lr_scheduler = copy.deepcopy(cfg_lr_scheduler)
        lr_scheduler_type = cfg_lr_scheduler.pop('type')
        if hasattr(lr_scheduler, lr_scheduler_type):
            if lr_scheduler_type == 'OneCycleLR':
                cfg_lr_scheduler['epochs'] = scheduler_cfg['epochs']
                cfg_lr_scheduler['steps_per_epoch'] = scheduler_cfg['steps_per_epoch']
            self.lr = getattr(lr_scheduler, lr_scheduler_type)(optimizer, **cfg_lr_scheduler)

        else:
            raise KeyError(f'lr_scheduler{lr_scheduler_type} is not found!!!')

    def scheduler_iteration_hook(self):
        if isinstance(self.lr, (CyclicLR, OneCycleLR, ExponentialLR)):
            self.lr.step()

    def scheduler_epoch_hook(self):
        if isinstance(self.lf, (StepLR, MultiStepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts)):
            self.lr.step()
