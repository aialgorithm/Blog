# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : AaronJny
# @Time    : 2019/12/30
# @Desc    :
import tensorflow as tf
from dataset import PoetryDataGenerator, tokenizer, poetry
from model import model
import settings
import utils


class Evaluate(tf.keras.callbacks.Callback):
    """
    在每个epoch训练完成后，保留最有权重，并随机生成settings.SHOW_NUM首古诗展示
    """

    def __init__(self):
        super().__init__()
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights(settings.BEST_MODEL_PATH)
        for i in range(settings.SHOW_NUM):
            print(utils.generate_random_poetry(tokenizer, model))


# 创建数据生成器
data_generator = PoetryDataGenerator(poetry, batch_size=settings.BATCH_SIZE)
# 开始训练
model.fit_generator(data_generator.forfit(), steps_per_epoch=data_generator.steps, epochs=settings.TRAIN_EPOCHS,
                    callbacks=[Evaluate()])
