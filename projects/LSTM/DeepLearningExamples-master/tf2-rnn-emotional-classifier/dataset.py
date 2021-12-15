# -*- coding: utf-8 -*-
# @File    : dataset.py
# @Author  : AaronJny
# @Time    : 2020/03/24
# @Desc    :
from collections import Counter
import math
import typing
import numpy as np
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


def load_data():
    """
    加载数据集
    """
    # 读取数据集
    with open(settings.DATASET_PATH, 'r', encoding='utf8') as f:
        lines = f.readlines()
    sentences = []
    labels = []
    # 逐行切分标签和句子
    for line in lines[1:]:
        label, *sentence = line.split(',')
        labels.append(int(label))
        sentences.append(','.join(sentence))
    return sentences, labels


def create_tokenizer(sentences: typing.List[str]) -> Tokenizer:
    """
    根据给定数据集，创建tokenizer

    Args:
        sentences: 评论数据句子的列表

    Returns:
        tokenizer
    """
    # 统计词频
    counter = Counter()
    for sentence in sentences:
        # 统计词频时，移除[CLS]和[SEP]字符
        counter.update(list(sentence))
    # 过滤低频词
    tokens_and_counts = [(token, count) for token, count in counter.items() if count >= settings.MIN_WORD_FREQUENCY]
    # 按词频倒序排列
    sorted_tokens_and_counts = sorted(tokens_and_counts, key=lambda x: -x[1])
    # 去掉词频，只保留token
    most_tokens = [token for token, count in sorted_tokens_and_counts]
    # 构建新词典
    tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + most_tokens
    token_dict = {}
    for token in tokens:
        token_dict[token] = len(token_dict)
    # 使用新词典构建分词器
    tokenizer = Tokenizer(token_dict)
    return tokenizer


def split_dataset(sentences: typing.List[str], labels: typing.List[int]) -> typing.Tuple:
    """
    划分数据集为训练集、开发集和测试集

    Args:
        sentences: 评论数据句子的列表
        labels: 正负标签的列表

    Returns:
        训练集，开发集，验证集
    """
    # 将数据变成[(sentence1,label1),(sentence2,label2),...]的格式
    data = list(zip(sentences, labels))
    # 混洗数据
    np.random.shuffle(data)
    # 计算每个数据集应该分配多少样例
    total = len(data)
    train_samples = int(total * settings.TRAIN_SPLIT)
    dev_samples = int(total * settings.DEV_SPLIT)
    train_data = data[:train_samples]
    dev_data = data[train_samples:train_samples + dev_samples]
    test_data = data[train_samples + dev_samples:]
    return train_data, dev_data, test_data


class WeiBoDataGenerator:

    def __init__(self, data, tokenizer, batch_size=32, random=True):
        # 数据集
        self.data = data
        self.tokenizer = tokenizer
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
        # 混洗数据
        if self.random:
            np.random.shuffle(self.data)
        # 迭代整个数据集，每次返回一个mini batch
        total = len(self.data)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_token_ids = []
            batch_labels = []
            # 逐样本进行处理
            for _sentence, _label in self.data[start:end]:
                token_ids = self.tokenizer.encode(_sentence)
                batch_token_ids.append(token_ids)
                batch_labels.append(_label)
            # 填充到相同长度
            batch_token_ids = self.sequence_padding(batch_token_ids, length=settings.MAX_LEN)
            batch_labels = np.reshape(np.array(batch_labels), (-1, 1))
            yield batch_token_ids, batch_labels
            del batch_token_ids
            del batch_labels

    def for_fit(self):
        """
        创建一个生成器，用于训练
        """
        # 死循环，当数据训练一个epoch之后，重新迭代数据
        while True:
            # 委托生成器
            yield from self.__iter__()


# 加载数据集
sentences, labels = load_data()
# 创建tokenizer
tokenizer = create_tokenizer(sentences)
# 划分数据集
train_data, dev_data, test_data = split_dataset(sentences, labels)
