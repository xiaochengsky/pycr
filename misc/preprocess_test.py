import sys
import os
import cv2
import numpy as np
import time
from glob import glob
from collections import Sequence
from tqdm import tqdm
import tensorflow as tf

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

_cv2_pad_to_str = {'constant': cv2.BORDER_CONSTANT,
                   'edge': cv2.BORDER_REPLICATE,
                   'reflect': cv2.BORDER_REFLECT_101,
                   'symmetric': cv2.BORDER_REFLECT
                   }
_cv2_interpolation_to_str = {'nearest': cv2.INTER_NEAREST,
                             'bilinear': cv2.INTER_LINEAR,
                             'area': cv2.INTER_AREA,
                             'bicubic': cv2.INTER_CUBIC,
                             'lanczos': cv2.INTER_LANCZOS4}
_cv2_interpolation_from_str = {v: k for k, v in _cv2_interpolation_to_str.items()}


def pad(image, padding, fill=0, padding_mode='constant'):
    if isinstance(padding, Sequence) and len(padding) == 4:
        pad_left = padding[0]
        pad_top = padding[1]
        pad_right = padding[2]
        pad_bottom = padding[3]
    if image.shape[2] == 1:
        return (cv2.copyMakeBorder(image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                   borderType=_cv2_pad_to_str[padding_mode], value=fill)[:, :, np.newaxis])
    else:
        return (cv2.copyMakeBorder(image, top=pad_top, bottom=pad_bottom, left=pad_left, right=pad_right,
                                   borderType=_cv2_pad_to_str[padding_mode], value=fill))


def rescale(image, output_size, interpolation=cv2.INTER_LINEAR):
    h, w = image.shape[:2]
    if isinstance(output_size, int):
        if h < w:
            new_h, new_w = output_size * h / w, output_size
            # new_h, new_w = output_size, output_size * w / h
        else:
            new_h, new_w = output_size, output_size * w / h
            # new_h, new_w = output_size * h / w, output_size
    else:
        new_h, new_w = output_size

    new_h, new_w = int(new_h), int(new_w)
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return image


def rescale_pad(image, output_size, interpolation=cv2.INTER_LINEAR, fill=0, padding_mode='constant'):
    image = rescale(image, output_size, interpolation)
    h, w = image.shape[:2]
    padding = [(output_size - w) // 2, (output_size - h) // 2, output_size - w - (output_size - w) // 2,
               output_size - h - (output_size - h) // 2]
    image = pad(image, padding, fill, padding_mode)
    return image


def preprocess(image_path, imgsz):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (1, 0, 2))
    image = rescale_pad(image, imgsz)
    image = image.astype(np.float32)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image /= 255.0
    image -= mean
    image /= std

    image = np.expand_dims(image, axis=0)
    return image


def infer(images_path, imgsz):
    image_nums = len(images_path)
    read_time = 0
    color_time = 0
    transpose_time = 0
    rescale_time = 0
    float32_time = 0
    div_time = 0
    norm_time = 0
    expand_time = 0
    start_time = time.time() * 1e3

    for image_path in tqdm(images_path):
        t0 = time.time() * 1e3
        image = cv2.imread(image_path)
        read_time += time.time() * 1e3 - t0

        t1 = time.time() * 1e3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color_time += time.time() * 1e3 - t1

        t2 = time.time() * 1e3
        image = np.transpose(image, (1, 0, 2))
        transpose_time += time.time() * 1e3 - t2

        t3 = time.time() * 1e3
        image = rescale_pad(image, imgsz)
        rescale_time += time.time() * 1e3 - t3

        t4 = time.time() * 1e3
        image = image.astype(np.float32)
        float32_time += time.time() * 1e3 - t4

        t5 = time.time() * 1e3
        image /= 255.0
        div_time += time.time() * 1e3 - t5

        t6 = time.time() * 1e3
        image -= mean
        image /= std
        norm_time += time.time() * 1e3 - t6

        t7 = time.time() * 1e3
        image = np.expand_dims(image, axis=0)
        expand_time += time.time() * 1e3 - t7

    end_time = time.time() * 1e3 - start_time

    print(f'time cost--->')
    print(f'read_time: {read_time / image_nums}:.3f')
    print(f'color_time: {color_time / image_nums}:.3f')
    print(f'transpose_time: {transpose_time / image_nums}:.3f')
    print(f'rescale_time: {rescale_time / image_nums}:.3f')
    print(f'float32_time: {float32_time / image_nums}:.3f')
    print(f'div_time: {div_time / image_nums}:.3f')
    print(f'norm_time: {norm_time / image_nums}:.3f')
    print(f'expand_time: {expand_time / image_nums}:.3f')
    print(f'total time: {end_time}:.3f')


if __name__ == '__main__':
    # models/tf2/mobileone_onnx_128pix.onnx/mobileone_onnx_128pix_float16.tflite
    image_root = sys.argv[1]
    imgsz = int(sys.argv[2])
    images_path = []
    with open(image_root, 'r') as f:
        # /media/mclab207/Datas/yc/ESFair2023/TrainingSet/G6/NV/0032069.jpg 3 6
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            images_path.append(line[0])

    infer(images_path, imgsz)
