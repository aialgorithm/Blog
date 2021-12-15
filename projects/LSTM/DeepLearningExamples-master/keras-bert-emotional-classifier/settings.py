# -*- coding: utf-8 -*-
# @File    : settings.py
# @Author  : AaronJny
# @Time    : 2020/03/24
# @Desc    :


# 数据集路径
DATASET_PATH = './weibo_senti_100k.csv'
# 预训练的模型参数
CONFIG_PATH = '/home/aaron/tools/chinese_L-12_H-768_A-12/bert_config.json'
CHECKPOINT_PATH = '/home/aaron/tools/chinese_L-12_H-768_A-12/bert_model.ckpt'
DICT_PATH = '/home/aaron/tools/chinese_L-12_H-768_A-12/vocab.txt'
# 允许的最小词频
MIN_WORD_FREQUENCY = 10
# 句子的最大长度
MAX_LEN = 128
# 训练集比例
TRAIN_SPLIT = 0.7
# 开发集比例
DEV_SPLIT = 0.15
# 学习率
LEARNING_RATE = 1e-6
# 训练epoch数
EPOCHS = 1
# mini batch大小
BATCH_SIZE = 16
# 权重存放路径
BEST_WEIGHTS_PATH = './best.weights'
