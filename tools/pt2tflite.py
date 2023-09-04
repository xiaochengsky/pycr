# -*- coding: utf-8 -*-
# @Time: 2023/6/22 上午8:54
# @Author: YANG.C
# @File: pt2tflite.py

import argparse
import os
import subprocess
import sys

sys.path.append('../pycr_tc')

import torch
import onnxruntime
from classification.utils.config import Config, DictAction
from classification.models.builder import CLASSIFIERS
from classification.datasets.builder import build_pipeline
from classification.utils.utils import load_checkpoints
from classification.models.backbones.mobileone import reparameterize_model
from classification.models.backbones.utils import replace_batchnorm


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


def onnx2tflite(model_name):
    if not os.path.exists('./models/tf2/'):
        os.makedirs('./models/tf2/')

    subprocess.run(f'onnx2tf -i {model_name} -o ./models/tf2/{model_name.split("/")[-1]}', shell=True)
    # subprocess.run(f'onnx2tf -i ./model/{model_name}_simplify.onnx -o ./model/tf2-sim/{model_name}',
    #                shell=True)


def pt2tflite(imgszs, weights_path):
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
    print(model)
    print(' = ' * 10)

    for img_size, weight_path in zip(imgszs, weights_path):
        # load model
        print(f'load model: {weight_path}, img_size: {img_size}')
        model = load_checkpoints(model, weight_path)
        model.eval()
        replace_batchnorm(model)
        reparameterize_model()
        ckpt = torch.load(weight_path, map_location='cpu')['model']
        ckpt.eval()

        imgsz = img_size
        dummy_input = torch.randn((1, 3, imgsz, imgsz)).float()
        if not os.path.exists('models'):
            os.makedirs('models')

        onnx_model_name = f'models/re_repvitm1_onnx_{str(imgsz)}pix.onnx'

        torch.onnx.export(model,
                          dummy_input,
                          onnx_model_name,
                          input_names=['input'],
                          output_names=['output'],
                          opset_version=12,
                          do_constant_folding=True)

        onnx_model = onnxruntime.InferenceSession(onnx_model_name)
        onnx_output = onnx_model.run(None, input_feed={'input': dummy_input.numpy()})
        model_output = model(dummy_input, None)
        ckpt_output = ckpt(dummy_input, None)
        re_model_output = re_model(dummy_input, None)
        print(
            f'export onnx successfully, onnx: {sum(sum(sum(onnx_output)))}, model: {model_output["classes"].sum()}, re_model: {re_model_output["classes"].sum()}, ckpt: {ckpt_output["classes"].sum()}')
        onnx2tflite(onnx_model_name)
        print(f'export tflite successfully')


if __name__ == '__main__':
    weights_path = [
        './logs/repvit/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_196pix/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_160pix/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_128pix/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_96pix/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_64pix/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_32pix/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_16pix/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_8pix/checkpoints/best_perf.pt',
        # './logs/MobileOneNet_baselines0_flip_IAA_lr3e-2_2pix/checkpoints/best_perf.pt',
    ]
    # imgszs = [224, 196, 160, 128, 96, 64, 32, 16, 8, 2]
    imgszs = [128, ]
    assert len(imgszs) == len(weights_path)
    pt2tflite(imgszs, weights_path)
