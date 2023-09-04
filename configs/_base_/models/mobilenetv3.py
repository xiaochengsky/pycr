# -*- coding: utf-8 -*-
# @Time: 2023/5/19 下午4:39
# @Author: YANG.C
# @File: mobilenetv3.py

tag = 'baseline_k4'

# model settings
BACKBONE_STAGE = 1
NECK_STAGE = 2
HEAD_STAGE = 4
BOTH_STAGE = 8
CLASSES_STAGE = 16

model = dict(
    type='Classifier',
    backbone=dict(
        type='mobilenetv3_small_050',
        pretrained=True,
        num_classes=0
    ),
    neck=dict(type='AdaptiveAvgPool2d', output_size=(1, 1), ),
    head=dict(type='IdentityHead', ),
    loss=[
        dict(type='CrossEntropy', position=HEAD_STAGE, in_feat=1024, num_classes=6, weight=1.0),
    ]
)
