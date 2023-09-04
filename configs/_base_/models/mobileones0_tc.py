# -*- coding: utf-8 -*-
# @Time: 2023/6/2 下午2:11
# @Author: YANG.C
# @File: mobileones0.py

# tag = 'baselines0'

# model settings
BACKBONE_STAGE = 1
NECK_STAGE = 2
HEAD_STAGE = 4
BOTH_STAGE = 8
CLASSES_STAGE = 16

model = dict(
    type='Classifier',
    backbone=dict(
        type='MobileOneNet',
        pretrained=True,
        num_classes=6,
        inference_mode=False,
        variant='s0',
        pretrained_path='/home/ycc/.cache/torch/hub/checkpoints/mobileone_s0_unfused.pth.tar'
    ),
    neck=dict(type='AdaptiveAvgPool2d', output_size=(1, 1), ),
    head=dict(type='BNNeckHead', in_feat=1024),
    loss=[
        dict(type='TripletLoss', position=NECK_STAGE, margin=0.6, weight=0.5),
        dict(type='ArcfaceLossDropout', position=HEAD_STAGE, in_feat=1024, num_classes=6, weight=1.0, dropout_rate=0.15),
    ]
)
