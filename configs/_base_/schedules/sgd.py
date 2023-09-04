# -*- coding: utf-8 -*-
# @Time : 2022/11/28 下午2:35
# @Author : YANG.C
# @File : sgd.py


# optimizer settings

optimizer = dict(type='SGD', lr=4e-4, weight_decay=1e-5)
warm_up = dict(iteration=1000, epoch=3, min_lr=4e-6, max_lr=4e-4, frozen_num_layer=8),
lr_scheduler = dict(type='ExponentialLR', gamma=0.99998)
