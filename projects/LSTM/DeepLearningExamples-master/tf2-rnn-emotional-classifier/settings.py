# -*- coding: utf-8 -*-
# @File    : settings.py
# @Author  : AaronJny
# @Time    : 2020/03/24
# @Desc    :


# 数据集路径
DATASET_PATH = './weibo_senti_100k.csv'
# 允许的最小词频
MIN_WORD_FREQUENCY = 10
# 句子的最大长度
MAX_LEN = 128
# 训练集比例
TRAIN_SPLIT = 0.7
# 开发集比例
DEV_SPLIT = 0.15
# 学习率
LEARNING_RATE = 1e-3
# 训练epoch数
EPOCHS = 4
# mini batch大小
BATCH_SIZE = 32
# 权重存放路径
BEST_WEIGHTS_PATH = './checkpoint/cp.ckpt'
