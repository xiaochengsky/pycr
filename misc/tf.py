# -*- coding: utf-8 -*-
# @Time: 2023/5/17 下午4:50
# @Author: YANG.C
# @File: tf.py

import os
import numpy as np
import subprocess
import onnx
import onnx_tf
import onnx2tf
import tensorflow as tf
# pip install onnx2tf
# pip install nvidia-pyindex
# pip install sng4onnx
# pip install onnx-graphsurgeon

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 加载 ONNX 模型
# onnx_model = onnx.load(f'./model/resnet18.onnx')
# tf_rep = onnx_tf.backend.prepare(onnx_model)
#
# subprocess.run(f'onnx2tf -i ./model/resnet18.onnx -o ./model/', shell=True)

model = tf.saved_model.load('./model')

input_data = tf.constant(1.0, shape=(1, 512, 512, 3), dtype=tf.float32)
output_data = model(input_data)
print(np.array(output_data).sum())

# 加载 TFLite 模型
interpreter = tf.lite.Interpreter(model_path=f'./model/resnet18_float32.tflite')
interpreter.allocate_tensors()

# 获取输入和输出张量索引
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 准备输入张量
input_shape = input_details[0]['shape']
input_data = np.array(input_data, dtype=np.float32)

# 将输入张量传递给模型
interpreter.set_tensor(input_details[0]['index'], input_data)

# 运行模型并获取输出张量
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

# 打印输出张量
print(np.array(output_data).sum())