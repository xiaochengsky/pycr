# -*- coding: utf-8 -*-
# @Time : 2022/12/2 上午11:51
# @Author : YANG.C
# @File : base_dataset.py


import copy
import os.path as osp
from os import PathLike
from abc import ABCMeta, abstractmethod

import cv2
import numpy as np
from torch.utils.data import Dataset

from .pipeline import Compose
from .sample import DistributedSampler


def expanduser(path):
    if isinstance(path, (str, PathLike)):
        return osp.expanduser(path)
    else:
        return path


class BaseDataset(Dataset):
    """Base dataset"""

    def __init__(self,
                 data_prefix,
                 pipeline,
                 classes=None):
        super(BaseDataset, self).__init__()
        self.data_prefix = expanduser(data_prefix)
        self.pipeline = Compose(pipeline)
        self.CLASSES = self.get_classes(classes)
        self.data_infos = self.load_annotations()

    @abstractmethod
    def load_annotations(self):
        pass

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.CLASSES)}

    @property
    def idx_to_class(self):
        return {i: _class for i, _class in enumerate(self.CLASSES)}

    def get_gt_labels(self):
        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels

    def get_cat_ids(self, idx):
        """Get category id by index."""
        return [int(self.data_infos[idx]['gt_label'])]

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        result = {}
        for k, v in data_info.items():
            if k == 'path':
                result['image'] = self.pipeline(data_info['path'])
            else:
                result[k] = v
        return result

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset."""
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, (tuple, list)):
            class_names = classes
        return class_names
