# -*- coding: utf-8 -*-
# @File    : model.py
# @Author  : AaronJny
# @Time    : 2020/01/01
# @Desc    :
import tensorflow as tf
from dataset import tokenizer

# 构建模型
model = tf.keras.Sequential([
    # 不定长度的输入
    tf.keras.layers.Input((None,)),
    # 词嵌入层
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128),
    # 第一个LSTM层，返回序列作为下一层的输入
    tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
    # 第二个LSTM层，返回序列作为下一层的输入
    tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True),
    # 对每一个时间点的输出都做softmax，预测下一个词的概率
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.vocab_size, activation='softmax')),
])

# 查看模型结构
model.summary()
# 配置优化器和损失函数
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy)
