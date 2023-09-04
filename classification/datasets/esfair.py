# -*- coding: utf-8 -*-
# @Time: 2023/5/18 下午7:53
# @Author: YANG.C
# @File: esfair.py

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module
class ESFair(BaseDataset):
    """ ESFair2023 Dataset
        ESFair
        |---train.txt           each line: /path/to/img.jpg(png,) class_id, group_id
        |---val.txt
    """

    CLASSES = [
        "BCC",  # 0
        "BKL",
        "MEL",
        "NV",
        "unknown",
        "VASC",  # 6
    ]

    def load_annotations(self):
        data_infos = []
        with open(self.data_prefix, 'r') as f:
            lines = f.readlines()
            for line in lines:
                info = {}
                line = line.strip('\n')
                # path/to/img.jpg class_id, group_id
                path = line.split(' ')[0]
                gt_label = int(line.split(' ')[1])
                gt_group = int(line.split(' ')[2])
                info['path'] = path
                info['gt_label'] = gt_label
                info['gt_group'] = gt_group
                data_infos.append(info)
            return data_infos
