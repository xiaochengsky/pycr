# -*- coding: utf-8 -*-
# @Time: 2023/7/10 下午11:10
# @Author: YANG.C
# @File: collect_fn.py

import torch


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
