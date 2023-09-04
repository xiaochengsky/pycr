# -*- coding: utf-8 -*-
# @Time : 2022/6/8 下午4:26
# @Author : YANG.C
# @File : filter.py


import os
from glob import glob
import cv2


def main():
    # root = '/home/yc/PyCharmProjects/datasets/VehiclePhaseDatasets'
    # images = glob(f'{root}/*/*.jpg')
    # images.extend(glob(f'{root}/*/*.png'))
    image_root = '/home/yc/PyCharmProjects/pycr/data/PAR-MASK/par_mask_train.txt'
    images = []
    with open(image_root, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n')
            image_name = line.split(' ')[0]
            images.append(image_name)

    image_cnts = 0
    error_cnts = 0
    correct_cnts = 0
    average_h, average_w = 0, 0
    for image in images:
        image_cnts += 1
        try:
            correct_cnts += 1
            img = cv2.imread(image)
            h, w, c = img.shape
            # print(img.shape)
            average_h += h
            average_w += w
        except Exception:
            error_cnts += 1
            print(image)
            # os.remove(image)

    average_h //= correct_cnts
    average_w //= correct_cnts

    print(f'img: {image_cnts}, cor: {correct_cnts}, err: {error_cnts}, average_h: {average_h}, average_w: {average_w}')


if __name__ == '__main__':
    main()
