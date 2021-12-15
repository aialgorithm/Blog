# -*- coding: utf-8 -*-
# @File    : dataset.py
# @Author  : AaronJny
# @Time    : 2019/12/30
# @Desc    : 构建数据集
from collections import Counter
import math
import numpy as np
import tensorflow as tf
import settings


class Tokenizer:
    """
    分词器
    """

    def __init__(self, token_dict):
        # 词->编号的映射
        self.token_dict = token_dict
        # 编号->词的映射
        self.token_dict_rev = {value: key for key, value in self.token_dict.items()}
        # 词汇表大小
        self.vocab_size = len(self.token_dict)

    def id_to_token(self, token_id):
        """
        给定一个编号，查找词汇表中对应的词
        :param token_id: 带查找词的编号
        :return: 编号对应的词
        """
        return self.token_dict_rev[token_id]

    def token_to_id(self, token):
        """
        给定一个词，查找它在词汇表中的编号
        未找到则返回低频词[UNK]的编号
        :param token: 带查找编号的词
        :return: 词的编号
        """
        return self.token_dict.get(token, self.token_dict['[UNK]'])

    def encode(self, tokens):
        """
        给定一个字符串s，在头尾分别加上标记开始和结束的特殊字符，并将它转成对应的编号序列
        :param tokens: 待编码字符串
        :return: 编号序列
        """
        # 加上开始标记
        token_ids = [self.token_to_id('[CLS]'), ]
        # 加入字符串编号序列
        for token in tokens:
            token_ids.append(self.token_to_id(token))
        # 加上结束标记
        token_ids.append(self.token_to_id('[SEP]'))
        return token_ids

    def decode(self, token_ids):
        """
        给定一个编号序列，将它解码成字符串
        :param token_ids: 待解码的编号序列
        :return: 解码出的字符串
        """
        # 起止标记字符特殊处理
        spec_tokens = {'[CLS]', '[SEP]'}
        # 保存解码出的字符的list
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token(token_id)
            if token in spec_tokens:
                continue
            tokens.append(token)
        # 拼接字符串
        return ''.join(tokens)


# 禁用词
disallowed_words = settings.DISALLOWED_WORDS
# 句子最大长度
max_len = settings.MAX_LEN
# 最小词频
min_word_frequency = settings.MIN_WORD_FREQUENCY
# mini batch 大小
batch_size = settings.BATCH_SIZE

# 加载数据集
with open(settings.DATASET_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    # 将冒号统一成相同格式
    lines = [line.replace('：', ':') for line in lines]
# 数据集列表
poetry = []
# 逐行处理读取到的数据
for line in lines:
    # 有且只能有一个冒号用来分割标题
    if line.count(':') != 1:
        continue
    # 后半部分不能包含禁止词
    __, last_part = line.split(':')
    ignore_flag = False
    for dis_word in disallowed_words:
        if dis_word in last_part:
            ignore_flag = True
            break
    if ignore_flag:
        continue
    # 长度不能超过最大长度
    if len(last_part) > max_len - 2:
        continue
    poetry.append(last_part.replace('\n', ''))

# 统计词频
counter = Counter()
for line in poetry:
    counter.update(line)
# 过滤掉低频词
_tokens = [(token, count) for token, count in counter.items() if count >= min_word_frequency]
# 按词频排序
_tokens = sorted(_tokens, key=lambda x: -x[1])
# 去掉词频，只保留词列表
_tokens = [token for token, count in _tokens]

# 将特殊词和数据集中的词拼接起来
_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + _tokens
# 创建词典 token->id映射关系
token_id_dict = dict(zip(_tokens, range(len(_tokens))))
# 使用新词典重新建立分词器
tokenizer = Tokenizer(token_id_dict)
# 混洗数据
np.random.shuffle(poetry)


class PoetryDataGenerator:
    """
    古诗数据集生成器
    """

    def __init__(self, data, random=False):
        # 数据集
        self.data = data
        # batch size
        self.batch_size = batch_size
        # 每个epoch迭代的步数
        self.steps = int(math.floor(len(self.data) / self.batch_size))
        # 每个epoch开始时是否随机混洗
        self.random = random

    def sequence_padding(self, data, length=None, padding=None):
        """
        将给定数据填充到相同长度
        :param data: 待填充数据
        :param length: 填充后的长度，不传递此参数则使用data中的最大长度
        :param padding: 用于填充的数据，不传递此参数则使用[PAD]的对应编号
        :return: 填充后的数据
        """
        # 计算填充长度
        if length is None:
            length = max(map(len, data))
        # 计算填充数据
        if padding is None:
            padding = tokenizer.token_to_id('[PAD]')
        # 开始填充
        outputs = []
        for line in data:
            padding_length = length - len(line)
            # 不足就进行填充
            if padding_length > 0:
                outputs.append(np.concatenate([line, [padding] * padding_length]))
            # 超过就进行截断
            else:
                outputs.append(line[:length])
        return np.array(outputs)

    def __len__(self):
        return self.steps

    def __iter__(self):
        total = len(self.data)
        # 是否随机混洗
        if self.random:
            np.random.shuffle(self.data)
        # 迭代一个epoch，每次yield一个batch
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_data = []
            # 逐一对古诗进行编码
            for single_data in self.data[start:end]:
                batch_data.append(tokenizer.encode(single_data))
            # 填充为相同长度
            batch_data = self.sequence_padding(batch_data)
            # yield x,y
            yield batch_data[:, :-1], tf.one_hot(batch_data[:, 1:], tokenizer.vocab_size)
            del batch_data

    def for_fit(self):
        """
        创建一个生成器，用于训练
        """
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            # 委托生成器
            yield from self.__iter__()