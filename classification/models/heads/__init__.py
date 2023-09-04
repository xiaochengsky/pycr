# -*- coding: utf-8 -*-
# @Time : 2022/11/29 下午5:25
# @Author : YANG.C
# @File : __init__.py

from .bnneck_head import BNNeckHead, BNNeckHeadDropout
from .identity_head import IdentityHead
from .reduction_head import ReductionHead

__all__ = [
    'BNNeckHead', 'BNNeckHeadDropout', "IdentityHead", "ReductionHead",
]
