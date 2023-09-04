# -*- coding: utf-8 -*-
# @Time: 2023/6/2 下午2:25
# @Author: YANG.C
# @File: mobileones0.py


tag = 'baselines0_flip_colorJitter'

_base_ = [
    '../_base_/models/mobileones0.py',
    '../_base_/datasets/esfair2023_fold1_flip_colorJitter.py',
    '../_base_/schedules/sgd_onecycle.py',
    '../_base_/default_runtime.py',
]
