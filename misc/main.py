# -*- coding: utf-8 -*-
# @Time: 2023/5/15 下午11:23
# @Author: YANG.C
# @File: main.py
import os.path
import sys
from tqdm import tqdm

import torch
import onnxruntime
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from pt2onnx import py2onnx
from onnx2tflite import onnx2tflite
from tensorflow2lite import tf2lite

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# estimate bias in accuracy and latency

def check_accuracy(model_name):
    C, imgsz = 3, 512
    cnts = 100
    model_types = ['pytorch', 'onnx', 'onnx-sim', 'tf', 'tf-sim', 'tflite16',
                   'tflite16-sim', 'tflite32', 'tflite32-sim']
    result = [0] * len(model_types)

    # N, C, H, W
    dummy_input = torch.ones(1, C, imgsz, imgsz).float()
    input_data = tf.constant(1.0, shape=(1, imgsz, imgsz, C), dtype=tf.float32)
    input_data = np.array(input_data, dtype=np.float32)

    # =============================================================================== #
    # test pytorch
    torch_model = torch.load(f'./model/{model_name}.pt', map_location='cpu')

    # =============================================================================== #
    # test onnx
    onnx_model = onnxruntime.InferenceSession(f'./model/{model_name}.onnx')
    # print(onnx_model.get_inputs(), onnx_model.get_outputs())

    # =============================================================================== #
    # test onnx_sim
    onnx_sim_model = onnxruntime.InferenceSession(f'./model/{model_name}_simplify.onnx')

    # =============================================================================== #
    # test tf2
    tf_model = tf.saved_model.load(f'./model/tf2/{model_name}')

    # =============================================================================== #
    # test tf2-simplify
    tf_sim_model = tf.saved_model.load(f'./model/tf2-sim/{model_name}')

    # =============================================================================== #
    # test tflite float16
    interpreter_tf16 = tf.lite.Interpreter(model_path=f'./model/tf2/{model_name}/{model_name}_float16.tflite')
    # interpreter_tf16.allocate_tensors()
    # input_details_tf16 = interpreter_tf16.get_input_details()
    # output_details_tf16 = interpreter_tf16.get_output_details()
    # interpreter_tf16.set_tensor(input_details_tf16[0]['index'], input_data)
    # interpreter_tf16.invoke()

    # =============================================================================== #
    # test tflite-sim float16
    interpreter_tfsim16 = tf.lite.Interpreter(
        model_path=f'./model/tf2-sim/{model_name}/{model_name}_simplify_float16.tflite')
    # interpreter_tfsim16.allocate_tensors()
    # input_details_tfsim16 = interpreter_tfsim16.get_input_details()
    # output_details_tfsim16 = interpreter_tfsim16.get_output_details()
    # interpreter_tfsim16.set_tensor(input_details_tfsim16[0]['index'], input_data)
    # interpreter_tfsim16.invoke()

    # =============================================================================== #
    # test tflite float32
    interpreter_tf32 = tf.lite.Interpreter(model_path=f'./model/tf2/{model_name}/{model_name}_float32.tflite')
    # interpreter_tf32.allocate_tensors()
    # input_details_tf32 = interpreter_tf32.get_input_details()
    # output_details_tf32 = interpreter_tf32.get_output_details()
    # interpreter_tf32.set_tensor(input_details_tf32[0]['index'], input_data)
    # interpreter_tf32.invoke()

    # =============================================================================== #
    # test tflite-sim float32
    interpreter_tfsim32 = tf.lite.Interpreter(
        model_path=f'./model/tf2-sim/{model_name}/{model_name}_simplify_float32.tflite')
    # interpreter_tfsim32.allocate_tensors()
    # input_details_tfsim32 = interpreter_tfsim32.get_input_details()
    # output_details_tfsim32 = interpreter_tfsim32.get_output_details()
    # interpreter_tfsim32.set_tensor(input_details_tfsim32[0]['index'], input_data)
    # interpreter_tfsim32.invoke()

    for i in tqdm(range(cnts)):
        dummy_input += i
        dummy_input /= 255.
        input_data += i
        input_data /= 255.

        interpreter_tf16.allocate_tensors()
        input_details_tf16 = interpreter_tf16.get_input_details()
        output_details_tf16 = interpreter_tf16.get_output_details()
        interpreter_tf16.set_tensor(input_details_tf16[0]['index'], input_data)
        interpreter_tf16.invoke()

        interpreter_tfsim16.allocate_tensors()
        input_details_tfsim16 = interpreter_tfsim16.get_input_details()
        output_details_tfsim16 = interpreter_tfsim16.get_output_details()
        interpreter_tfsim16.set_tensor(input_details_tfsim16[0]['index'], input_data)
        interpreter_tfsim16.invoke()

        interpreter_tf32.allocate_tensors()
        input_details_tf32 = interpreter_tf32.get_input_details()
        output_details_tf32 = interpreter_tf32.get_output_details()
        interpreter_tf32.set_tensor(input_details_tf32[0]['index'], input_data)
        interpreter_tf32.invoke()

        interpreter_tfsim32.allocate_tensors()
        input_details_tfsim32 = interpreter_tfsim32.get_input_details()
        output_details_tfsim32 = interpreter_tfsim32.get_output_details()
        interpreter_tfsim32.set_tensor(input_details_tfsim32[0]['index'], input_data)
        interpreter_tfsim32.invoke()

        result[0] = torch_model(dummy_input).sum().item()
        result[1] = np.array(onnx_model.run(None, input_feed={'input': dummy_input.numpy()})).sum().item()
        result[2] = np.array(onnx_sim_model.run(None, input_feed={'input': dummy_input.numpy()})).sum().item()
        result[3] = np.array(tf_model(input_data)).sum().item()
        result[4] = np.array(tf_sim_model(input_data)).sum().item()
        result[5] = np.array(interpreter_tf16.get_tensor(output_details_tf16[0]['index'])).sum().item()
        result[6] = np.array(interpreter_tfsim16.get_tensor(output_details_tfsim16[0]['index'])).sum().item()
        result[7] = np.array(interpreter_tf32.get_tensor(output_details_tf32[0]['index'])).sum().item()
        result[8] = np.array(interpreter_tfsim32.get_tensor(output_details_tfsim32[0]['index'])).sum().item()
        for i in range(len(result)):
            result[len(result) - i - 1] -= result[0]

        print(result)

    # loss_str = ''
    # for i in range(len(model_types)):
    #     loss_str += model_types[i] + ': '
    #     loss_str += str((result[i] - result[0]) / cnts) + ' '
    #
    # print(loss_str)


def check_latency(model_name):
    pass


if __name__ == '__main__':
    # must be in torchvision.models
    model_name = sys.argv[1]
    if not os.path.exists('model'):
        os.makedirs('model')

    # py2onnx
    print('<<<<<< pytorch ---> onnx/onnx_sim >>>>>>')
    py2onnx(model_name)

    # onnx2tflite
    print('<<<<<< onnx/sim ---> tf/tflite >>>>>>')
    onnx2tflite(model_name)

    # check_accuracy(model_name)
