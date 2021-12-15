# -*- coding: utf-8 -*-
# @File    : eval.py
# @Author  : AaronJny
# @Time    : 2020/03/25
# @Desc    : 自己输入句子测试模型是否有效
from dataset import tokenizer
from models import model
import settings

# 加载训练好的参数
model.load_weights(settings.BEST_WEIGHTS_PATH)

print('启动验证程序！')
while True:
    try:
        sentence = input('请输入一句话，模型将判断其情绪倾向：')
        token_ids, segment_ids = tokenizer.encode(sentence)
        output = model.predict([[token_ids, ], [segment_ids, ]])[0][0]
        if output > 0.5:
            print('正面情绪！')
        else:
            print('负面情绪！')
    except KeyboardInterrupt:
        print('结束程序！')
        break

"""
请输入一句话，模型将判断其情绪倾向：虽然没有买到想要的东西，但我并不沮丧           
正面情绪！
请输入一句话，模型将判断其情绪倾向：没有买到想要的东西， 有点沮丧   
负面情绪！
请输入一句话，模型将判断其情绪倾向：书挺好的，就是贵了点
正面情绪！
请输入一句话，模型将判断其情绪倾向：书的确不错，但也太贵了
负面情绪！
"""
