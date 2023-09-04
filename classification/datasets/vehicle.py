# -*- coding: utf-8 -*-
# @Time : 2022/12/2 下午2:34
# @Author : YANG.C
# @File : vehicle.py


import os.path as osp
from typing import Optional
import numpy as np

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module
class Vehicle(BaseDataset):
    """ VehiclePhaseTwo Dataset.

        Vehicle
        |---train.txt       each lines: /path/to/img.jpg(png,) 1
        |---val.txt
    """
    CLASSES = [
        "suv",  # 0, suv
        "car",  # 1, 小轿车
        'taxi',  # 2, 出租车
        'buses',  # 3, 公交巴士
        'bus',  # -4, 中小巴士
        'coach',  # -5, 大巴
        'doubleDeckerBus',  # -6, 双层巴士
        'lorry',  # 7, 货车
        'largeLorry',  # 8, 大货车
        'tanker',  # 9, 罐车
        'transport',  # -10, 危险运输车
        'roller',  # -11, 压路车
        'trailer',  # -12, 挂车
        'excavator',  # -13, 挖掘机
        'dumper',  # 14, 泥头车
        'sprinkler',  # 15, 洒水车
        'forklift',  # -16, 铲车
        'crane',  # -17, 起重机
        'fireEngine',  # 18, 消防车
        'ambulance',  # 19, 救护车
        'eco',  # 20, 环卫车
        'paver',  # 21, 摊铺机
        'pile',  # 22, 打桩机
        'carCrane',  # 23, 汽车吊
        'bicycle',  # 24, 自行车
        'electric',  # 25, 电动车
        'engineering',  # 26, 工程车
        'pedestrian',  # 27, 行人
        'other'  # 28, 其它
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
