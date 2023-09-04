# -*- coding: utf-8 -*-
# @Time: 2023/8/1 下午3:43
# @Author: YANG.C
# @File: regnety_160_tc.py

# model settings
BACKBONE_STAGE = 1
NECK_STAGE = 2
HEAD_STAGE = 4
BOTH_STAGE = 8
CLASSES_STAGE = 16

model = dict(
    type='Classifier',
    backbone=dict(
        type='repvit_m1',
        pretrained=True,
    ),
    neck=dict(type='AdaptiveAvgPool2d', output_size=(1, 1), ),
    head=dict(type='BNNeckHead', in_feat=384, ),
    loss=[
        dict(type='TripletLoss', position=NECK_STAGE, margin=0.6, weight=0.5),
        dict(type='ArcfaceLossDropout', position=HEAD_STAGE, in_feat=384, num_classes=6, weight=1.0,
             dropout_rate=0.15),
    ]
)

