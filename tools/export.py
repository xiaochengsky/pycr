# -*- coding: utf-8 -*-
# @Time : 2022/12/7 上午11:51
# @Author : YANG.C
# @File : export.py

import sys
import os
import argparse

sys.path.append('../pycr_tc')

import cv2
import numpy as np

import torch
import onnxruntime

from classification.utils.config import Config, DictAction
from classification.models.builder import CLASSIFIERS
from classification.datasets.builder import build_pipeline
from classification.utils.utils import load_checkpoints, load_resume_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--model', help='build model'
    )
    return parser.parse_args()


def main(ckpt):
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint

    # build model
    model_config = cfg.model
    model_type = model_config.pop('type')
    model = CLASSIFIERS[model_type](model_config)
    model.eval()
    model = load_resume_model(model, cfg.checkpoint)
    model.eval()
    imgsz = cfg.imgsz
    dummy_input = torch.ones(1, 3, imgsz, imgsz).float()
    # dummy_input += torch.tensor([0, 1, 2])
    pipeline = build_pipeline(cfg.data.test.pipeline)

    torch.onnx.export(model,
                      dummy_input,
                      './models/best_pref.onnx',
                      input_names=['input'],
                      output_names=['output'],
                      opset_version=12,
                      do_constant_folding=True)

    onnx_model = onnxruntime.InferenceSession('./models/best_pref.onnx')
    onnx_output = onnx_model.run(None, input_feed={'input': dummy_input.numpy()})
    model_output = model(dummy_input, None)

    print(f'onnx: {onnx_output}')
    print(f'model: {model_output["classes"]}')


if __name__ == '__main__':
    ckpt = sys.argv[1]
    main(ckpt)
