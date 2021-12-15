# -*- coding: utf-8 -*-
# @File    : settings.py
# @Author  : AaronJny
# @Time    : 2019/12/25
# @Desc    :

# 禁用词，包含如下字符的唐诗将被忽略
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
# 句子最大长度
MAX_LEN = 64
# 最小词频
MIN_WORD_FREQUENCY = 8
# 训练的batch size
BATCH_SIZE = 16
# 数据集路径
DATASET_PATH = './poetry.txt'
# 每个epoch训练完成后，随机生成SHOW_NUM首古诗作为展示
SHOW_NUM = 5
# 共训练多少个epoch
TRAIN_EPOCHS = 20
# 最佳权重保存路径
BEST_MODEL_PATH = './best_model.h5'
