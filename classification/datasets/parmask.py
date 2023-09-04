# -*- coding: utf-8 -*-
# @Time : 2022/12/2 下午2:34
# @Author : YANG.C
# @File : vehicle.py


import os.path as osp
from typing import Optional
import numpy as np
import torch

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module
class ParMask(BaseDataset):
    """ VehiclePhaseTwo Dataset.

        Vehicle
        |---train.txt       each lines: /path/to/img.jpg(png,) 1,1,2,3,4,4,5,5,6,7,45,45...
        |---val.txt
    """
    CLASSES = ['male', 'femal', 'child', 'youth', 'middle_aged', 'elderly', 'short_sleeve', 'long_sleeve',
               'upper_with_logo', 'upper_no_logo', 'upper_black', 'upper_white', 'upper_red', 'upper_green',
               'upper_blue', 'upper_yellow', 'upper_mix', 'shorts', 'trousers', 'short_skirt', 'long_skirt',
               'lower_black', 'lower_white', 'lower_red', 'lower_green', 'lower_blue', 'lower_yellow', 'lower_mix',
               'no_bag', 'backpack', 'shoulder_bag', 'diagonal_bag', 'front', 'side', 'back', 'hat', 'no_hat', 'glass',
               'no_glass', 'short_hair', 'long_hair', 'handheld', 'no_handheld', 'mask', 'orientation', 'invalid']

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
                all_label = line.split(' ')[1]
                all_label = all_label.split(',')
                info['path'] = path
                info['gt_label'] = torch.tensor([int(i) for i in all_label])
                data_infos.append(info)
        return data_infos
