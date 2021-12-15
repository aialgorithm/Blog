# -*- coding: utf-8 -*-
# @File    : models.py
# @Author  : AaronJny
# @Time    : 2020/03/24
# @Desc    :
from bert4keras.models import build_transformer_model
import keras
from keras import backend as K
from dataset import keep_tokens
import settings

# 加载bert模型
bert_model = build_transformer_model(config_path=settings.CONFIG_PATH, checkpoint_path=settings.CHECKPOINT_PATH,
                                     keep_tokens=keep_tokens)
# 提取[CLS]位置的输出，在此任务中可以理解为句向量
output = keras.layers.Lambda(lambda x: x[:, 0])(bert_model.output)
# sigmoid做二分类
output = keras.layers.Dense(1, activation=keras.activations.sigmoid)(output)
# 构建最终模型
model = keras.Model(bert_model.input, output)


def recall_m(y_true, y_pred):
    """
    计算召回率
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    计算精确率
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
    计算f1值
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# 选择Adam作为优化器，准确率、精准率、召回率、f1值作为监控指标
model.compile(keras.optimizers.Adam(settings.LEARNING_RATE), loss=keras.losses.binary_crossentropy,
              metrics=['accuracy', f1_m, precision_m, recall_m])
