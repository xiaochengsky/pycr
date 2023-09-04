# -*- coding: utf-8 -*-
# @Time : 2022/12/16 下午12:45
# @Author : YANG.C
# @File : encoder.py


import os
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)


def main():
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

    mapVType = dict()
    for i in range(len(CLASSES)):
        mapVType[CLASSES[i]] = i

    # mapVType = dict(zip(vTypeMap.values(), vTypeMap.keys()))

    data_root = '/home/yc/PyCharmProjects/datasets/Huawei'

    ft = open('data/huawei/hw_train_label.txt', 'w')
    fv = open('data/huawei/hw_val_label.txt', 'w')

    vtype = os.listdir(data_root)
    for vt in vtype:
        if vt in mapVType.keys():
            vtype_idx = mapVType[vt]
        else:
            continue
        images = os.listdir(os.path.join(data_root, vt))
        random.shuffle(images)
        idx = 0
        for image in images:
            if idx < len(images) * 0.85:
                ft.write(
                    os.path.join(data_root, vt, image) + ' ' + str(vtype_idx) + ' ' + '\n')
            else:
                fv.write(
                    os.path.join(data_root, vt, image) + ' ' + str(vtype_idx) + ' ' + '\n')
            idx += 1
    ft.close()
    fv.close()


if __name__ == '__main__':
    main()
