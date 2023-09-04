# -*- coding: utf-8 -*-
# @Time: 2023/6/2 下午2:25
# @Author: YANG.C
# @File: mobileones0.py


tag = 'baselines0_many_shift_nocolor_lr1e-3_96pix_300e_arcdrop_pretrain_alldata_ema_tc_kd'
distillation = True

_base_ = [
    '../_base_/models/mobileones0_s2_tc.py',
    '../_base_/datasets/esfair2023_fold1_many_96pix_alldata_tc.py',
    '../_base_/schedules/sgd_onecycle.py',
    '../_base_/default_runtime.py',
]
