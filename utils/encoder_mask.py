# -*- coding: utf-8 -*-
# @Time : 2022/12/16 下午3:48
# @Author : XXX
# @File : encoder_mask.py


from glob import glob
import os
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)

images_path = glob(f'/home/yc/PyCharmProjects/datasets/Mask/*/*/*.png')
random.shuffle(images_path)

ft = open('data/huawei/hw_mask_train_label.txt', 'w')
fv = open('data/huawei/hw_mask_val_label.txt', 'w')

# WithMask 0, WithoutMask 1

for image_path in images_path:
    label = 0 if 'WithMask' in image_path else 1
    if random.random() < 0.9:
        ft.write(f'{image_path}' + ' ' + str(label) + '\n')
    else:
        fv.write(f'{image_path}' + ' ' + str(label) + '\n')
