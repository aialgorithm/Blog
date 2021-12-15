# -*- coding: utf-8 -*-
# @Date         : 2020-10-30
# @Author       : AaronJny
# @LastEditTime : 2021-01-18
# @FilePath     : /DeepLearningExamples/tf2-rnn-emotional-classifier/train.py
# @Desc         :

# -*- coding: utf-8 -*-
# @File    : train.py
# @Author  : AaronJny
# @Time    : 2020/03/26
# @Desc    :
import tensorflow as tf
from dataset import train_data, dev_data, test_data
from dataset import tokenizer
from dataset import WeiBoDataGenerator
from models import model
import settings

train_generator = WeiBoDataGenerator(
    train_data, tokenizer, settings.BATCH_SIZE)
dev_generator = WeiBoDataGenerator(dev_data, tokenizer, settings.BATCH_SIZE)
test_generator = WeiBoDataGenerator(test_data, tokenizer, settings.BATCH_SIZE)

# 自动保存模型
checkpoint = tf.keras.callbacks.ModelCheckpoint(settings.BEST_WEIGHTS_PATH, monitor='val_f1_m', save_best_only=True,
                                                save_weights_only=True, mode='max')
# 训练
model.fit_generator(train_generator.for_fit(), steps_per_epoch=train_generator.steps, epochs=settings.EPOCHS,
                    validation_data=dev_generator.for_fit(), validation_steps=dev_generator.steps,
                    callbacks=[checkpoint, ])
# 测试
print('测试集结果：')
loss, accuracy, f1_score, precision, recall = model.evaluate_generator(test_generator.for_fit(),
                                                                       steps=test_generator.steps)
print('loss =', loss)
print('accuracy =', accuracy)
print('f1 score =', f1_score)
print('precision =', precision)
print('recall =', recall)
