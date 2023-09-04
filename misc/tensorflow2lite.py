# -*- coding: utf-8 -*-
# @Time: 2023/5/15 下午11:20
# @Author: YANG.C
# @File: tensorflow2lite.py

import tensorflow as tf


def tf2lite(model_name):
    tf_name = f'./model/{model_name}.pb'
    tf_name_simplify = f'./model/{model_name}_simplify.pb'

    tflite_name = f'./model/{model_name}.tflite'
    tflite_name_simplify = f'./model/{model_name}_simplify.tflite'

    tf_model = tf.keras.models.load_model(tf_name)
    tf_model_simplify = tf.keras.models.load_model(tf_name_simplify)

    print(tf_model.summary())

    # 定义签名键
    inputs = {'input': tf_model.input}
    outputs = {'output': tf_model.output}

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_name, signature_keys=inputs)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(tflite_name, 'wb') as f:
        f.write(tf_lite_model)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_name_simplify)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(tflite_name_simplify, 'wb') as f:
        f.write(tf_lite_model)


if __name__ == '__main__':
    model_name = 'resnet18'
    tf2lite(model_name)
