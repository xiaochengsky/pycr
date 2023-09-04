# -*- coding: utf-8 -*-
# @Time : 2022/11/29 下午5:27
# @Author : XXX
# @File : test_model.py
import torch

import models
from models.builder import build_backbone, build_neck, build_head, build_loss

from models.classifiers import Classifier

BACKBONE_STAGE = 1
NECK_STAGE = 2
HEAD_STAGE = 4
BOTH_STAGE = 8
CLASSES_STAGE = 16

if __name__ == '__main__':
    model = dict(
        type='Classifier',
        backbone=dict(
            type='resnet101',
            pretrained=False,
            num_classes=0
        ),
        neck=dict(type='AdaptiveAvgPool2d', output_size=(1, 1), ),
        head=dict(type='IdentityHead', ),
        loss=[
            dict(type='CrossEntropy', position=HEAD_STAGE, in_feat=2048, num_classes=3, weight=1.0),
        ]
    )

    neck = build_neck(model['neck'])
    print(neck)
    head = build_head(model['head'])
    print(head)
    loss = build_loss(model['loss'])
    print(loss)

    model = Classifier(backbone=model['backbone'], neck=model['neck'], head=model['head'], loss=model['loss'])
    print(model)
    torch.save(model.eval(), 'model.pt')

    inputs = torch.randn((1, 3, 224, 224))
    logits = torch.LongTensor([1.])
    output = model(inputs, gt_labels=logits)

    # save_model = torch.load('model.pt')
