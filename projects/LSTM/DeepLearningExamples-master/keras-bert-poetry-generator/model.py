# -*- coding: utf-8 -*-
# @File    : model.py
# @Author  : AaronJny
# @Time    : 2019/12/25
# @Desc    :
from bert4keras.models import build_transformer_model
import tensorflow as tf
from dataset import keep_words
import settings

model = build_transformer_model(settings.CONFIG_PATH, settings.CHECKPOINT_PATH, application='lm', keep_tokens=keep_words)

model.summary()

# loss fun，交叉熵
# 输入的数据，从第二个字符开始，可以作为正确的目标结果(输入是没有经过one-hot编码的)
y_true = model.input[0][:, 1:]
# 目标mask
y_mask = model.get_layer('Embedding-Token').output_mask[:, 1:]
y_mask = tf.cast(y_mask, tf.float32)
# 预测结果，到倒数第二个（包括）时结束
y_pred = model.output[:, :-1]
cross_entropy = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = tf.reduce_sum(cross_entropy * y_mask) / tf.reduce_sum(y_mask)
model.add_loss(cross_entropy)
model.compile(tf.keras.optimizers.Adam(1e-5))
