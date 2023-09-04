# -*- coding: utf-8 -*-
# @Time: 2023/5/19 上午9:50
# @Author: YANG.C
# @File: preprocessing.py

import os
import sys
from glob import glob
import numpy as np
import random

CLASSES = [
    "BCC",  # 0
    "BKL",
    "MEL",
    "NV",
    "unknown",
    "VASC",  # 6
]


def k_fold(root, save, k=5):
    seed = 42
    np.random.seed(seed)
    random.seed(seed)

    class2idx = {_class: i for i, _class in enumerate(CLASSES)}
    idx2class = {i: _class for i, _class in enumerate(CLASSES)}
    class_sum = [0] * len(class2idx)

    ft = [None for i in range(k)]
    fv = [None for i in range(k)]
    if not os.path.exists(save):
        os.makedirs(save)

    for id in range(0, k):
        if not os.path.exists(f'{save}/fold_{id}'):
            os.makedirs(f'{save}/fold_{id}')
        ft[id] = open(f'{save}/fold_{id}/esfair_train_fold{id}.txt', 'w')
        fv[id] = open(f'{save}/fold_{id}/esfair_val_fold{id}.txt', 'w')

    groups = list(filter(lambda x: 'G' in x, os.listdir(f'{root}')))
    class_group_sum = {}
    print(f'groups: {groups}')

    # each groups
    for g in groups:
        g_num = g.split('G')[1]
        # print(f'group: {g}, g_num: {g_num}')
        class_group_sum[g] = {_class: 0 for i, _class in enumerate(CLASSES)}
        classes = os.listdir(f'{root}/{g}')

        if '.DS_Store' in classes:
            classes.remove('.DS_Store')

        # print(classes, class_group_sum)
        # group/class
        for c in classes:

            class_id = class2idx[c]
            class_sum[class_id] += len(os.listdir(f'{root}/{g}/{c}'))
            class_group_sum[g][c] += len(os.listdir(f'{root}/{g}/{c}'))

            # group/class/img.jpg
            image_paths = glob(f'{root}/{g}/{c}/*.jpg')
            random.shuffle(image_paths)
            unit = len(image_paths) // k

            for id in range(k):  # 0
                for i in range(len(image_paths)):
                    info = image_paths[i] + ' ' + str(class_id) + ' ' + str(g_num) + '\n'

                    if len(image_paths) >= k:  # enough
                        pos = i / unit  #
                        if int(pos) == id:  # pos: unit*id~unit*(id+1))
                            fv[id].write(info)
                        else:  # pos: unit_left
                            ft[id].write(info)

                    if len(image_paths) < k:  # not enough
                        ratio = 0.5
                        if i < len(image_paths) * ratio:
                            ft[id].write(info)
                        else:
                            fv[id].write(info)

    # release
    for id in range(0, k):
        ft[id].close()
        fv[id].close()

    # for g, v in class_group_sum.items():
    #     for k, n in v.items():
    #         print(f'{g}, {k}: {n} ')

    for i in range(len(class_sum)):
        print(f'{idx2class[i]}: {class_sum[i]}, avg {class_sum[i] // k}')

    # check
    for i in range(k):
        train_classes = {}
        val_classes = {}
        ft_path = f'{save}/fold_{i}/esfair_train_fold{i}.txt'
        fv_path = f'{save}/fold_{i}/esfair_val_fold{i}.txt'
        ft_lines = []
        fv_lines = []
        with open(ft_path, 'r') as ft:
            # /home/ycc/PycharmProjects/datasets/ESFaire2023/TrainingSet/G8/unknown/6356522.jpg 4 8
            lines = ft.readlines()
            ft_lines.extend(lines.copy())
            for line in lines:
                line = line.strip('\n')
                class_id = int(line.split(' ')[1])
                class_name = idx2class[class_id]
                if class_name not in train_classes:
                    train_classes[class_name] = 0
                train_classes[idx2class[class_id]] += 1
        with open(fv_path, 'r') as fv:
            # /home/ycc/PycharmProjects/datasets/ESFaire2023/TrainingSet/G8/unknown/6356522.jpg 4 8
            lines = fv.readlines()
            fv_lines.extend(lines.copy())
            for line in lines:
                line = line.strip('\n')
                class_id = int(line.split(' ')[1])
                class_name = idx2class[class_id]
                if class_name not in val_classes:
                    val_classes[class_name] = 0
                val_classes[idx2class[class_id]] += 1
        print(f'pre-shuffle fold: {i}:')
        print(dict(sorted(train_classes.items(), key=lambda x: x[0])))
        print(dict(sorted(val_classes.items(), key=lambda x: x[0])))

        # reshuffle
        train_classes = {}
        val_classes = {}
        random.shuffle(ft_lines)
        random.shuffle(fv_lines)
        with open(ft_path, 'w') as ft:
            for line in ft_lines:
                ft.write(line)
        with open(fv_path, 'w') as fv:
            for line in fv_lines:
                fv.write(line)

        with open(ft_path, 'r') as ft:
            # /home/ycc/PycharmProjects/datasets/ESFaire2023/TrainingSet/G8/unknown/6356522.jpg 4 8
            lines = ft.readlines()
            for line in lines:
                line = line.strip('\n')
                class_id = int(line.split(' ')[1])
                class_name = idx2class[class_id]
                if class_name not in train_classes:
                    train_classes[class_name] = 0
                train_classes[idx2class[class_id]] += 1
        with open(fv_path, 'r') as fv:
            # /home/ycc/PycharmProjects/datasets/ESFaire2023/TrainingSet/G8/unknown/6356522.jpg 4 8
            lines = fv.readlines()
            for line in lines:
                line = line.strip('\n')
                class_id = int(line.split(' ')[1])
                class_name = idx2class[class_id]
                if class_name not in val_classes:
                    val_classes[class_name] = 0
                val_classes[idx2class[class_id]] += 1
        print(f'post-shuffle fold: {i}:')
        print(dict(sorted(train_classes.items(), key=lambda x: x[0])))
        print(dict(sorted(val_classes.items(), key=lambda x: x[0])))


if __name__ == '__main__':
    """
    root = '/home/ycc/PycharmProjects/datasets/ESFaire2023/TrainingSet'
    save = '/home/ycc/PycharmProjects/pycr/data/esfair'
    """
    root = sys.argv[1]
    save = sys.argv[2]
    k = 5
    # fair k fold
    k_fold(root, save, k)
