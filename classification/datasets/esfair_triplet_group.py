# -*- coding: utf-8 -*-
# @Time: 2023/6/25 下午11:42
# @Author: YANG.C
# @File: esfair_triplet.py

import copy
import random
from PIL import Image
from collections import defaultdict

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# from .base_dataset import BaseDataset
# from .builder import DATASETS
#
#
# @DATASETS.register_module
class ESFairTriplet(Dataset):
    """ ESFair2023 Dataset
        ESFair
        |---train.txt           each line: /path/to/img.jpg(png,) class_id, group_id
        |---val.txt
    """

    def __init__(self,
                 data_prefix,
                 pipeline,
                 cls_per_batch=6,
                 img_per_cls=4,
                 group_per_batch=4,
                 step_all=100):
        super(ESFairTriplet, self).__init__()

        self.read_order = []
        self.CLASSES = [
            "BCC",  # 0
            "BKL",
            "MEL",
            "NV",
            "unknown",
            "VASC",  # 5
        ]

        self.data_prefix = data_prefix
        self.pipeline = pipeline
        self.cls_per_batch = cls_per_batch
        self.img_per_cls = img_per_cls
        self.data_infos = self.load_annotations()

        self.group_minority = 10
        self.group2cls2path = {}
        # group2cls2path[10][0('BCC')] = [1.jpg, 2.jpg]
        for di in self.data_infos:
            img_path = di['path']
            gt_label = int(di['gt_label'])  # [0 - 5]
            gt_group = int(di['gt_group'])
            if gt_group not in self.group2cls2path:
                self.group2cls2path[gt_group] = {}
                for i in range(len(self.CLASSES)):
                    self.group2cls2path[gt_group][i] = []

            self.group2cls2path[gt_group][gt_label].append(img_path)

        # upper sample minority each epochs
        self.step_all = step_all

    def shuffle(self):
        print('shuffle manual')
        self.read_order = []
        step_data = []
        tmp = []
        for group in self.group2cls2path.keys():  # 4
            # TODO random select categories
            for cls in self.group2cls2path[group].keys():  # 6

                if len(self.group2cls2path[group][cls]) > self.step_all * self.img_per_cls:
                    samples = random.sample(self.group2cls2path[group][cls], k=self.step_all * self.img_per_cls)
                else:
                    samples = random.choices(self.group2cls2path[group][cls], k=self.step_all * self.img_per_cls)

                for j in range(len(samples)):  # 300 * 4
                    info = {'path': samples[j], 'gt_label': int(cls), 'gt_group': int(group)}
                    tmp.append(info)

                random.shuffle(tmp)
                step_data.append(tmp)  # 24 * [300 * 4]
                tmp = []

        for s in range(self.step_all):
            tmp = []
            for i in range(len(step_data)):  # 4 * 6
                tmp.extend(step_data[i][s * self.img_per_cls: (s + 1) * self.img_per_cls])
            random.shuffle(tmp)
            self.read_order.append(tmp)
        print(self.read_order[0])

    def __len__(self):
        return self.step_all

    def __getitem__(self, step):
        print(f'getitem, step: {step}')
        if step > self.step_all - 1:
            print('step train out of size')
            return
        print(f'getitem, len(self.read_order): {len(self.read_order)}')
        result = []
        for idx in range(len(self.read_order[step])):
            # print(f'getitem, len(self.read_order[step]): {len(self.read_order[step])}')
            # result.append(self.prepare_triplet_data(self.read_order[step][idx]))
            result.append(self.prepare_triplet_data(step, idx))
        return result

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

    def prepare_data(self, idx):
        data_info = copy.deepcopy(self.data_infos[idx])
        result = {}
        for k, v in data_info.items():
            if k == 'path':
                # result['image'] = self.pipeline(data_info['path'])
                result['image'] = data_info['path']
                result['path'] = data_info['path']
            else:
                result[k] = v
        return result

    def prepare_triplet_data(self, step, idx):
        data_info = copy.deepcopy(self.read_order[step][idx])
        result = {}
        # print(data_info)

        for k, v in data_info.items():
            if k == 'path':
                img = Image.open(data_info['path'])
                result['image'] = self.pipeline(img)
                result['path'] = data_info['path']
                # result['image'] = self.pipeline(data_info['path'])
                # result['image'] = data_info['path']

            else:
                result[k] = v
        return result


def collect_function(batch_list):
    image_list = list()
    label_list = list()
    group_list = list()
    path_list = list()

    for i in range(len(batch_list[0])):
        bl = batch_list[0][i]
        image = bl['image'].unsqueeze(0)
        label = torch.tensor(bl['gt_label']).unsqueeze(0)
        group = torch.tensor(bl['gt_group']).unsqueeze(0)
        path = bl['path']

        image_list.append(image)
        label_list.append(label)
        group_list.append(group)
        path_list.append(path)
    return {
        'image': torch.cat(image_list, dim=0),
        'gt_label': torch.cat(label_list, dim=0),
        'gt_group': torch.cat(group_list, dim=0),
        'path': path_list
    }


if __name__ == '__main__':
    pipeline = transforms.Compose([transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = ESFairTriplet('/home/ycc/PycharmProjects/pycr/data/esfair/fold_1/esfair_train_fold1.txt',
                            pipeline=pipeline,
                            )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collect_function,
        # worker_init_fn=init_fn,
        # **kwargs
    )

    print(data_loader, len(data_loader))

    for epoch in range(5):
        data_loader.dataset.shuffle()
        pbar = enumerate(data_loader)
        for i, batch in pbar:
            if i == 0:
                print(f'i: {i}')
                # print(batch['image'].shape)
                # print(batch['gt_label'])
                # print(batch['gt_group'])
                # print(batch['path'])
            # exit(0)
