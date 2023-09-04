# -*- coding: utf-8 -*-
# @Time : 2022/12/1 下午3:39
# @Author : YANG.C
# @File : classifier.py
import torch

from .base import BaseClassifier
from ..builder import CLASSIFIERS, build_backbone, build_neck, build_head, build_loss

BACKBONE_STAGE = 1
NECK_STAGE = 2
HEAD_STAGE = 4
BOTH_STAGE = 8
CLASSES_STAGE = 16


@CLASSIFIERS.register_module
class Classifier(BaseClassifier):
    def __init__(self,
                 # backbone,
                 # neck=None,
                 # head=None,
                 # loss=None,
                 model_config
                 ):
        super(Classifier, self).__init__()

        backbone = model_config.backbone
        neck = model_config.neck
        head = model_config.head
        loss = model_config.loss

        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if head is not None:
            self.head = build_head(head)

        if loss is not None:
            self.losses = build_loss(loss)

        self.loss_positions = []

        for i in range(len(self.losses)):
            pos_ = list(self.losses.keys())[i]
            loss_ = list(self.losses.values())[i]
            self.loss_positions.append(pos_)
            exec(f'self.losses{i} = loss_')  # insert into model, matching `model.cuda` and some operations

        self.tta = None

    def extract_feat(self, imgs, stage=HEAD_STAGE):
        assert stage in [BACKBONE_STAGE, NECK_STAGE, HEAD_STAGE, BOTH_STAGE], f'Invalid output stage "{stage}",  \
        please choose from "BACKBONE_SATGE", "NECK_STAGE", "HEAD_STAGE", "BOTH_STAGE"'

        feats = {}

        backbone_feat = self.backbone(imgs)
        feats[BACKBONE_STAGE] = backbone_feat if stage == BACKBONE_STAGE or BOTH_STAGE else None
        if stage == BACKBONE_STAGE:
            return feats

        neck_feat = self.neck(backbone_feat)
        feats[NECK_STAGE] = neck_feat if stage == NECK_STAGE or BOTH_STAGE else None
        if stage == NECK_STAGE:
            return feats

        head_feat = self.head(neck_feat)
        feats[HEAD_STAGE] = head_feat if stage == HEAD_STAGE or BOTH_STAGE else None
        if stage == HEAD_STAGE:
            return feats
        if stage == BOTH_STAGE:
            return feats

    def forward_train(self, imgs, gt_labels, teacher_outputs, **kwargs):
        """Forward computation during training"""
        if self.tta is not None:
            pass
        feats = self.extract_feat(imgs, stage=BOTH_STAGE)

        # compute losses
        total_loss = {}
        avg_loss = []
        for pos, loss_func in self.losses.items():
            if feats[pos] is not None:
                if pos == HEAD_STAGE:
                    aloss, loss = loss_func(feats[pos], gt_labels, teacher_outputs)
                    avg_loss.extend(loss)
                    total_loss[pos] = aloss
                else:
                    loss = loss_func(feats[pos], gt_labels)
                    avg_loss.append(loss)
                    total_loss[pos] = loss

        # total_loss['total_loss'] = sum(avg_loss) / len(avg_loss)
        # total_loss = sum(avg_loss) / len(avg_loss)
        total_loss = sum(avg_loss)
        return total_loss, avg_loss

    def simple_test(self, img, **kwargs):
        """Test without TTA"""
        head_feat = self.extract_feat(img, stage=HEAD_STAGE)[HEAD_STAGE]
        # res = self.losses[HEAD_STAGE].simple_test(head_feat, **kwargs)
        res = self.losses[HEAD_STAGE](head_feat, **kwargs)
        # return {
        #     'features': head_feat,
        #     'classes': res
        # }
        return {
            # 'features': head_feat,
            'classes': res
        }
