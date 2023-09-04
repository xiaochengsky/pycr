# -*- coding: utf-8 -*-
# @Time: 2023/6/16 上午11:51
# @Author: YANG.C
# @File: inference_test.py

import sys
import os
import cv2
import numpy as np
import time
from glob import glob
from collections import Sequence
from tqdm import tqdm
import tensorflow as tf

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


def infer(weight_path, images_path, imgsz):
    t0 = int(round(time.time() * 1000))
    # model init
    interpreter = tf.lite.Interpreter(model_path=weight_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Allocate output array
    t1 = int(round(time.time() * 1e3))  # load model time cost

    preprocess_timecost = 0
    inference_timecost = 0
    postprocess_timecost = 0

    # obtain image path
    image_nums = len(images_path)
    for image_path in tqdm(images_path):
        # preprocess
        pt0 = time.time() * 1e3
        image = preprocess(image_path, imgsz)
        preprocess_timecost += time.time() * 1e3 - pt0

        # Set input tensor value
        it0 = time.time() * 1e3
        interpreter.set_tensor(input_details[0]["index"], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        inference_timecost += time.time() * 1e3 - it0

        # TODO: batch process
        ct0 = time.time() * 1e3
        class_index = np.argmax(output_data)
        postprocess_timecost += time.time() * 1e3 - ct0

    t2 = time.time() * 1e3
    total_timecost = t2 - t1
    avg_preprocess_timecost = preprocess_timecost / image_nums
    avg_inference_timecost = inference_timecost / image_nums
    avg_postprocess_timecost = postprocess_timecost / image_nums
    print(f'weight: {weight_path}, imgsz: {imgsz}, image nums: {image_nums} '
          f'total time cost: {total_timecost:.3f}, avg_preprocess_timecost: '
          f'{avg_preprocess_timecost:.3f}, avg_inference_timecost: {avg_inference_timecost:.3f}, '
          f'avg_postprocess_timecost: {avg_postprocess_timecost:.3f}, load model time cost: {(t1 - t0):.3f}')
    print('\n\n')


if __name__ == '__main__':
    # models/tf2/mobileone_onnx_128pix.onnx/mobileone_onnx_128pix_float16.tflite
    weight_root = sys.argv[1]
    image_root = sys.argv[2]
    print(f'weight_root: {weight_root}, image_root: {image_root}')

    images_path = []
    with open(image_root, 'r') as f:
        # /media/mclab207/Datas/yc/ESFair2023/TrainingSet/G6/NV/0032069.jpg 3 6
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            images_path.append(line[0])

    weight_types = os.listdir(weight_root)
    for wt in weight_types:
        imgsz = int(wt.split('.')[0].split('_')[-1].rstrip('pix'))
        weights_path = glob(f'{weight_root}/{wt}/*.tflite')
        for weight_path in weights_path:
            infer(weight_path, images_path, imgsz)
