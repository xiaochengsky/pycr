# -*- coding: utf-8 -*-
# @Time : 2022/11/28 下午2:48
# @Author : YANG.C
# @File : resnet50_b16x8_vp.py


_base_ = [
    '../_base_/models/resnet50_mask.py',
    '../_base_/datasets/mask_b8.py',
    '../_base_/schedules/sgd_onecycle.py',
    '../_base_/default_runtime.py',
]


