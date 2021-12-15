# -*- coding: utf-8 -*-
# @File    : models.py
# @Author  : AaronJny
# @Time    : 2020/03/26
# @Desc    :
import tensorflow as tf
from dataset import tokenizer
import settings

model = tf.keras.Sequential([
    tf.keras.layers.Input((None,)),
    tf.keras.layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=128),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
])

model.summary()

EPSILON = 1e-07


def recall_m(y_true, y_pred):
    """
    计算召回率
    """
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + EPSILON)
    return recall


def precision_m(y_true, y_pred):
    """
    计算精确率
    """
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + EPSILON)
    return precision


def f1_m(y_true, y_pred):
    """
    计算f1值
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + EPSILON))


model.compile(optimizer=tf.keras.optimizers.Adam(settings.LEARNING_RATE), loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy', f1_m, precision_m, recall_m])
