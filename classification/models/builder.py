# -*- coding: utf-8 -*-
# @Time : 2022/11/29 上午10:57
# @Author : YANG.C
# @File : builder.py

import sys
import os
from copy import deepcopy

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import timm

from classification.utils.registry import Registry

BACKBONES = Registry('backbones')
NECKS = Registry('necks')
HEADS = Registry('heads')
LOSSES = Registry('losses')
CLASSIFIERS = Registry('classifiers')

BACKBONE_STAGE = 1
NECK_STAGE = 2
HEAD_STAGE = 4
BOTH_STAGE = 8
CLASSES_STAGE = 16


def build_backbone(cfg):
    """Build backbone."""
    # must be covered in TIMM
    cfg = deepcopy(cfg)
    backbone_type = cfg.pop('type')
    # except MobileOneNet
    if 'mobileonenet' in backbone_type.lower():
        backbone = BACKBONES[backbone_type](**cfg)
    else:
        backbone = timm.create_model(backbone_type, **cfg)
        backbone = nn.Sequential(
            *list(backbone.children())[:-2]) if 'repvit' not in backbone_type.lower() else nn.Sequential(
            *list(backbone.children())[:-1])
    return backbone


def build_neck(cfg):
    """Build neck."""
    cfg = deepcopy(cfg)
    neck_type = cfg.pop('type')
    return NECKS[neck_type](**cfg)


def build_head(cfg):
    """Build head."""
    cfg = deepcopy(cfg)
    head_type = cfg.pop('type')
    return HEADS[head_type](**cfg)


def build_loss(cfg):
    """Build loss."""
    cfg = deepcopy(cfg)
    positions = []
    losses = {}

    for cg in cfg:
        cg = deepcopy(cg)
        loss_type = cg.pop('type')
        pos = cg.pop('position')
        # TODO, multi-types loss in same pos
        losses[pos] = LOSSES[loss_type](**cg)
        positions.append(pos)

    return losses
    # return losses[CLASSES_STAGE]


def build_model(cfg):
    """Build model."""
    cfg = deepcopy(cfg)
    backbone = build_backbone(cfg['backbone'])
    neck = build_neck(cfg['neck'])
    head = build_head(cfg['head'])
    loss = build_loss(cfg['loss'])
    return {
        'backbone': backbone,
        'neck': neck,
        'head': head,
        'loss': loss
    }
