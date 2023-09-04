# -*- coding: utf-8 -*-
# @Time : 2022/11/29 下午3:30
# @Author : YANG.C
# @File : test.py

import sys
import os
import argparse

sys.path.append('../pycr')

import cv2
import numpy as np

import torch
import onnxruntime

from classification.utils.config import Config, DictAction
from classification.models.builder import CLASSIFIERS
from classification.datasets.builder import build_pipeline
from classification.api.inference import inference_model


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


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint

    inference_model(cfg)


if __name__ == '__main__':
    main()
