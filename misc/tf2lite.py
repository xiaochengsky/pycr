# -*- coding: utf-8 -*-
# @Time: 2023/5/18 上午11:05
# @Author: YANG.C
# @File: tf2lite.py

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

# 加载ResNet50模型（不包括顶部分类器）
resnet_model = ResNet50(weights='imagenet', include_top=False)

# 打印模型摘要
print(resnet_model.summary())

# 定义一个输入张量，我们使用ImageNet数据集中图像的形状作为输入形状，因为ResNet50是在该数据集上进行了训练。
input_shape = (512, 512, 3)
input_tensor = tf.keras.layers.Input(shape=input_shape)

# 使用输入张量创建模型，以便在保存模型时可以提供输入形状
model = tf.keras.models.Model(inputs=[input_tensor], outputs=[resnet_model(input_tensor)])

# 转换为TFLite格式并保存到文件
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("resnet50.tflite", "wb").write(tflite_model)
