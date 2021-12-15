# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : AaronJny
# @Time    : 2020/01/01
# @Desc    :
import tensorflow as tf
from dataset import PoetryDataGenerator, poetry, tokenizer
from model import model
import settings
import utils


class Evaluate(tf.keras.callbacks.Callback):
    """
    在每个epoch训练完成后，保留最优权重，并随机生成settings.SHOW_NUM首古诗展示
    """

    def __init__(self):
        super().__init__()
        # 给loss赋一个较大的初始值
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epoch训练完成后调用
        # 如果当前loss更低，就保存当前模型参数
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save(settings.BEST_MODEL_PATH)
        # 随机生成几首古体诗测试，查看训练效果
        print()
        for i in range(settings.SHOW_NUM):
            print(utils.generate_random_poetry(tokenizer, model))


# 创建数据集
data_generator = PoetryDataGenerator(poetry, random=True)
# 开始训练
model.fit_generator(data_generator.for_fit(), steps_per_epoch=data_generator.steps, epochs=settings.TRAIN_EPOCHS,
                    callbacks=[Evaluate()])
