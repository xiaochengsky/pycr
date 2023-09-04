# -*- coding: utf-8 -*-
# @Time : 2022/11/29 下午5:25
# @Author : YANG.C
# @File : __init__.py

from .pooling import AdaptiveAvgMaxPool2d, FastGlobalAvgPool2d, Flatten, GeneralizedMeanPooling, \
    GeneralizedMeanPoolingP, SoftPool2d, AdaptiveAvgPool2d

__all__ = [
    'AdaptiveAvgMaxPool2d', 'FastGlobalAvgPool2d', 'Flatten', 'GeneralizedMeanPooling',
    'GeneralizedMeanPoolingP', 'SoftPool2d', 'AdaptiveAvgPool2d',
]
