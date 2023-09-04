# -*- coding: utf-8 -*-
# @Time : 2022/12/6 上午9:45
# @Author : YANG.C
# @File : inference.py

import copy
import random
import time
import warnings
import os
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

import torch

random.seed(42)
from classification.utils.config import Config, DictAction
from classification.models.builder import CLASSIFIERS
from classification.datasets.builder import build_pipeline, build_dataset
from classification.utils.utils import load_checkpoints
from tools.train import log_config


def inference_model(cfg):
    logs = './logs'
    tag = cfg.tag
    base_dir = f'{logs}/{cfg.model.backbone.type}_{tag}'

    # build model
    model_config = cfg.model
    model_type = model_config.pop('type')
    model = CLASSIFIERS[model_type](model_config)
    device = 'cuda:0'
    model = load_checkpoints(model, cfg.checkpoint)
    model.eval()
    model.to(device)
    imgsz = cfg.imgsz
    pipeline = build_pipeline(cfg.data.test.pipeline)

    # test dirs
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 1
    val_dataset = build_dataset(cfg.data.val)
    idx_to_class = val_dataset.idx_to_class
    classes = val_dataset.CLASSES
    pbar = tqdm(val_dataset)
    gap = 150
    gt_text_location_x, gt_text_location_y = 50, 300
    pred_text_location_x, pred_text_location_y = 400, 300

    if not os.path.exists(f'{base_dir}/vis'):
        os.makedirs(f'{base_dir}/vis')
    print(f'save dir: f{base_dir}/vis')

    for i, batch in enumerate(pbar):
        if i > 10:
            break
        if random.random() < 1:
            path = batch['path']
            ori_img = cv2.imread(path)
            ori_img = cv2.resize(ori_img, (ori_img.shape[1] // 2, ori_img.shape[0] // 2))
            targets = batch['gt_label']
            targets = targets.to(device)

            img = pipeline(path)
            img = img.to(device, non_blocking=True).float()
            img = torch.unsqueeze(img, 0)
            preds = model(img)['classes']
            preds = torch.sigmoid(preds) >= 0.5

            new_img = np.zeros((800, 800, 3)).astype(ori_img.dtype)
            h, w, c = ori_img.shape
            new_img[gap:gap + h, gap:gap + w, :] = ori_img
            pil_img = Image.fromarray(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            font = ImageFont.truetype('/home/yc/Documents/simhei.ttf', 50, encoding='utf-8')
            new_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            preds_list = []
            gts_list = []
            for j in range(len(preds[0])):
                if j % 10 == 0:
                    gt_text_location_x = 50
                    gt_text_location_y += 50
                    pred_text_location_x = 400
                    pred_text_location_y += 50
                if preds[0][j].item() == 1:
                    preds_list.append(str(idx_to_class[j]))
                    draw.text((pred_text_location_x, pred_text_location_y), str(idx_to_class[j]), (255, 0, 0), font)
                if targets[j].item() == 1:
                    gts_list.append(str(idx_to_class[j]))
                    draw.text((gt_text_location_x, gt_text_location_y), str(idx_to_class[j]), (255, 0, 0), font)
                gt_text_location_x += 50
                pred_text_location_x += 50

            print(f'img_path: {path}')
            print(preds_list)
            print(gts_list)
            new_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{base_dir}/vis/{path.split("/")[-1]}', new_img)

            # exit(0)
