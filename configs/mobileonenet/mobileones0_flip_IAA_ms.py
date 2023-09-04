# -*- coding: utf-8 -*-
# @Time: 2023/6/2 下午2:25
# @Author: YANG.C
# @File: mobileones0.py


tag = 'baselines0_flip_IAA_ms_lr1e-3'

_base_ = [
    '../_base_/models/mobileones0.py',
    '../_base_/datasets/esfair2023_fold1_flip_IAA_ms.py',
    '../_base_/schedules/sgd_onecycle.py',
    '../_base_/default_runtime.py',
]
