# -*- coding: utf-8 -*-
# @Date         : 2020-10-30
# @Author       : AaronJny
# @LastEditTime : 2021-01-18
# @FilePath     : /DeepLearningExamples/keras-bert-emotional-classifier/train.py
# @Desc         :
import keras
from dataset import MyDataGenerator
from dataset import tokenizer
from dataset import train_data, dev_data, test_data
from models import model
import settings

# 创建数据集迭代器
train_generator = MyDataGenerator(train_data, tokenizer, settings.BATCH_SIZE)
dev_generator = MyDataGenerator(dev_data, tokenizer, settings.BATCH_SIZE)
test_generator = MyDataGenerator(test_data, tokenizer, settings.BATCH_SIZE)

# 设置checkpoint，自动保存模型
checkpoint = keras.callbacks.ModelCheckpoint(
    settings.BEST_WEIGHTS_PATH, monitor='val_f1_m', save_best_only=True, mode='max')

# 训练
model.fit_generator(train_generator.forfit(), steps_per_epoch=train_generator.steps, epochs=settings.EPOCHS,
                    validation_data=dev_generator.forfit(), validation_steps=dev_generator.steps,
                    callbacks=[checkpoint, ])
# 测试
print('测试集结果：')
loss, accuracy, f1_score, precision, recall = model.evaluate_generator(test_generator.forfit(),
                                                                       steps=test_generator.steps)
print('loss =', loss)
print('accuracy =', accuracy)
print('f1 score =', f1_score)
print('precision =', precision)
print('recall =', recall)
