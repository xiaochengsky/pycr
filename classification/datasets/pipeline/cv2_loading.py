# -*- coding: utf-8 -*-
# @Time : 2022/12/7 下午12:01
# @Author : YANG.C
# @File : cv2_loading.py

import cv2
import jpeg4py as jpeg

from ..builder import PIPELINES


@PIPELINES.register_module
class LoadImageFromFile(object):
    """Load an image from filename"""

    def __init__(self):
        pass

    def __call__(self, result):
        image_path = result
        # image = cv2.imread(image_path)
        # image = image[:, :, ::-1].copy()  # BGR2RGB
        image = jpeg.JPEG(image_path).decode()
        return image


@PIPELINES.register_module
class LoadPartImageFromFile(object):
    def __init__(self):
        pass

    def __call__(self, result, part=0.1):
        image_path = result
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        start_h, start_w = int(h * part), int(w * part)
        end_h, end_w = h - int(h * part), w - int(w * part)
        image = image[:, :, ::-1].copy()
        image = image[start_h: end_h, start_w: end_w, :]
        return image

