# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : AaronJny
# @Time    : 2019/12/30
# @Desc    :
import numpy as np
import settings


def generate_random_poetry(tokenizer, model, s=''):
    """
    随机生成一首诗
    :param tokenizer: 分词器
    :param model: 用于生成古诗的模型
    :param s: 用于生成古诗的起始字符串，默认为空串
    :return: 一个字符串，表示一首古诗
    """
    # 将初始字符串转成token
    token_ids, segment_ids = tokenizer.encode(s)
    # 去掉结束标记[SEP]
    token_ids = token_ids[:-1]
    # 去掉对应的segment
    segment_ids = segment_ids[:-1]
    # 保存所有预测出的id的列表
    target_ids = []
    while len(token_ids) + len(target_ids) < settings.MAX_LEN:
        # 将给定的初始token和预测出的token扩展到一起，作为输入
        _target_ids = token_ids + target_ids
        _segment_ids = segment_ids + [0 for _ in target_ids]
        # 进行预测，只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[PAD][UNK][CLS]的概率分布
        _probas = model.predict([[_target_ids, ], [_segment_ids, ]])[0, -1, 3:]
        # 此时_probas的shape为(3188,)，其中3188是我们当前数据下的词汇表大小-3
        # print(_probas.shape)
        # 按照出现概率，对所有token倒序排列
        p_args = _probas.argsort()[::-1][:100]
        # 排列后的概率顺序
        p = _probas[p_args]
        # 先对概率归一
        p = p / sum(p)
        # 再按照预测出的概率，随机选择一个词作为预测结果
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index] + 3
        # 保存
        target_ids.append(target)
        if target == 3:
            break
    return tokenizer.decode(token_ids + target_ids)


def generate_acrostic(tokenizer, model, head):
    """
    随机生成一首藏头诗
    :param tokenizer: 分词器
    :param model: 用于生成古诗的模型
    :param head: 藏头诗的头
    :return: 一个字符串，表示一首古诗
    """
    # 使用空串初始化token_ids，加入[CLS]
    token_ids, segment_ids = tokenizer.encode('')
    token_ids = token_ids[:-1]
    segment_ids = segment_ids[:-1]
    # 标点符号，这里简单的只把逗号和句号作为标点
    punctuations = ['，', '。']
    punctuation_ids = {tokenizer.token_to_id(token) for token in punctuations}
    # 缓存生成的诗的list
    poetry = []
    # 对于藏头诗中的每一个字，都生成一个短句
    for ch in head:
        # 先记录下这个字
        poetry.append(ch)
        # 将藏头诗的字符转成token id
        token_id = tokenizer.token_to_id(ch)
        # 加入到列表中去
        token_ids.append(token_id)
        segment_ids.append(0)
        # 开始生成一个短句
        while True:
            # 进行预测，只保留第一个样例（我们输入的样例数只有1）的、最后一个token的预测的、不包含[PAD][UNK][CLS]的概率分布
            _probas = model.predict([[token_ids, ], [segment_ids, ]])[0, -1, 3:]
            # 按照出现概率，对所有token倒序排列
            p_args = _probas.argsort()[::-1][:100]
            # 排列后的概率顺序
            p = _probas[p_args]
            # 先对概率归一
            p = p / sum(p)
            # 再按照预测出的概率，随机选择一个词作为预测结果
            target_index = np.random.choice(len(p), p=p)
            target = p_args[target_index] + 3
            # 保存
            token_ids.append(target)
            segment_ids.append(0)
            # 只有不是特殊字符时，才保存到poetry里面去
            if target > 3:
                poetry.append(tokenizer.id_to_token(target))
            if target in punctuation_ids:
                break
    return ''.join(poetry)
