# -*- coding: utf-8 -*-
# @Time : 2022/12/1 下午5:11
# @Author : YANG.C
# @File : identity_head.py


import torch
import torch.nn as nn

from ..builder import HEADS


@HEADS.register_module
class IdentityHead(nn.Module):
    def __init__(self):
        super(IdentityHead, self).__init__()

    def forward(self, features):
        return features[..., 0, 0]


if __name__ == '__main__':
    n = torch.randn(3, 3, 1, 1)
    m = IdentityHead()
    print(m(n).shape)
