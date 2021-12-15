# -*- coding: utf-8 -*-
"""
本项目源自
github.com/AaronJny/DeepLearningExamples/tree/master/tf2-rnn-poetry-generator
做了些修改
"""

import tensorflow as tf
from dataset import tokenizer
import settings
import utils

# 加载训练好的模型
model = tf.keras.models.load_model(settings.BEST_MODEL_PATH)

while True:
    poem_type = input('\n**输入序号选择功能:**\n[1]随机生成一首诗\n[2]给出诗的开头，生成后面的诗\n[3]生成藏头诗\n[4]退出\n')
    if poem_type == '1':
        # 随机生成一首诗
        print(utils.generate_random_poetry(tokenizer, model))
    elif poem_type == '2':
        keywords = input('输入:\n')
        # 给出部分信息的情况下，随机生成剩余部分
        print(utils.generate_random_poetry(tokenizer, model, s=keywords))
    elif poem_type == '3':
        keywords = input('输入:\n')
        # 生成藏头诗
        print(utils.generate_acrostic(tokenizer, model, head=keywords))
    elif poem_type == '4':
        break

