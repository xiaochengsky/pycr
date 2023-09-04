# -*- coding: utf-8 -*-
# @Time : 2022/12/6 上午9:51
# @Author : YANG.C
# @File : optimizer.py

import copy

from torch import optim


def build_optimizer(cfg_optimizer, model):
    cfg_optimizer = copy.deepcopy(cfg_optimizer)
    optimizer_type = cfg_optimizer.pop('type')
    if hasattr(optim, optimizer_type):
        optimizer = getattr(optim, optimizer_type)(model.parameters(), **cfg_optimizer)
        return optimizer
    else:
        raise KeyError(f'optimizer {optimizer_type} is not found!')
