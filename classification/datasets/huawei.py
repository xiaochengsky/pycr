# -*- coding: utf-8 -*-
# @Time : 2022/12/2 下午2:34
# @Author : YANG.C
# @File : huawei.py


import os.path as osp
from typing import Optional
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module
class Huawei(BaseDataset):
    """ Huawei Dataset.

        Huawei
        |---train.txt       each lines: /path/to/img.jpg(png,) 1
        |---val.txt
    """
    CLASSES = [
        "bus",  # 0, 包车
        "buses",  # 1, 班车
        "lorry-Cover",  # 2, 货车
        "lorry-UnConver",  # 3,
        "dumper6",  # 4, 泥头车
        "danger-slogan",  # 5, 危险车辆
        "danger-noslogon"  # 6, 

        # color
        "unknown",  # 7
        "white",  # 8
        "gray",  # 9
        "yellow",  # 10
        "pink",  # 12
        "red",  # 12
        "purple",  # 13
        "green",  # 14
        "brown",  # 15
        "black",  # 16
        "orange",  # 17
        "cyan",  # 18
        "silver",  # 19
        "champagne",  # 20
        "blue",  # 21
    ]

    # def __init__(self, data_prefix: str):
    #     super(Vehicle, self).__init__()
    #     self.data_prefix = data_prefix

    def load_annotations(self):
        data_infos = []
        with open(self.data_prefix, 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = {}
                line = line.strip('\n')
                # path/to/img.jpg 1
                path = line.split(' ')[0]
                gt_label = int(line.split(' ')[1])
                info['path'] = path
                info['gt_label'] = int(gt_label)
                data_infos.append(info)
        return data_infos
