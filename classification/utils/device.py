# -*- coding: utf-8 -*-
# @Time : 2022/12/2 下午4:04
# @Author : YANG.C
# @File : device.py

import torch


def auto_select_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
