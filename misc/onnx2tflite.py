# -*- coding: utf-8 -*-
# @Time: 2023/5/15 下午11:10
# @Author: YANG.C
# @File: onnx2tensorflow2.py


import subprocess


def onnx2tflite(model_name):
    subprocess.run(f'onnx2tf -i ./model/{model_name}.onnx -o ./model/tf2/{model_name}', shell=True)
    subprocess.run(f'onnx2tf -i ./model/{model_name}_simplify.onnx -o ./model/tf2-sim/{model_name}',
                   shell=True)


if __name__ == '__main__':
    model_name = 'resnet18'
    onnx2tflite(model_name)
