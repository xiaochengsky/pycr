# -*- coding: utf-8 -*-
# @Time : 2022/11/25 下午4:26
# @Author : YANG.C
# @File : resnet50.py

#
tag = 'PAR'

# model settings
BACKBONE_STAGE = 1
NECK_STAGE = 2
HEAD_STAGE = 4
BOTH_STAGE = 8
CLASSES_STAGE = 16

model = dict(
    type='Classifier',
    backbone=dict(
        type='resnet50',
        pretrained=True,
        num_classes=0
    ),
    neck=dict(type='AdaptiveAvgPool2d', output_size=(1, 1), ),
    head=dict(type='IdentityHead', ),
    loss=[
        dict(type='MultiLabelBCE', position=HEAD_STAGE, in_feat=2048, num_classes=46, weight=1.0, invalid=45),
    ]
)
