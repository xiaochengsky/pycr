# -*- coding: utf-8 -*-
# @Time : 2022/11/29 下午4:49
# @Author : YANG.C
# @File : __init__.py


from .cross_entropy import CrossEntropy, CrossEntropySplitFC, LabelSmoothCrossEntropy
from .arcface import ArcfaceLoss
from .focalloss import FocalLoss
from .triplet import TripletLoss
from .multilabel_bce import MultiLabelBCE
from .distillation import DistillationLoss

__all__ = [
    'CrossEntropy', 'ArcfaceLoss', 'FocalLoss', 'MultiLabelBCE', 'CrossEntropySplitFC', 'LabelSmoothCrossEntropy',
    'TripletLoss', 'DistillationLoss',
]
