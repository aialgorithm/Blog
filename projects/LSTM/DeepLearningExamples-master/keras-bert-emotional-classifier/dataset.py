# -*- coding: utf-8 -*-
# @File    : dataset.py
# @Author  : AaronJny
# @Time    : 2020/03/24
# @Desc    :
from collections import Counter
import typing
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import DataGenerator, sequence_padding
import numpy as np
import settings


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


def create_tokenizer(sentences: typing.List[str]) -> typing.Tuple[Tokenizer, typing.List]:
    """
    根据新的数据集，精简词表，重新创建tokenizer

    Args:
        sentences: 评论数据句子的列表

    Returns:
        tokenizer,keep_tokens
    """
    # 加载下载的词表
    _token_dict = load_vocab(settings.DICT_PATH)
    _tokenizer = Tokenizer(_token_dict, do_lower_case=True)

    # 统计词频
    counter = Counter()
    for sentence in sentences:
        _tokens = _tokenizer.tokenize(sentence)
        # 统计词频时，移除[CLS]和[SEP]字符
        counter.update(_tokens[1:-1])
    # 过滤低频词
    tokens_and_counts = [(token, count) for token, count in counter.items() if count >= settings.MIN_WORD_FREQUENCY]
    # 按词频倒序排列
    sorted_tokens_and_counts = sorted(tokens_and_counts, key=lambda x: -x[1])
    # 去掉词频，只保留token
    most_tokens = [token for token, count in sorted_tokens_and_counts]
    # 构建新词典
    tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]'] + most_tokens
    keep_tokens = []
    token_dict = {}
    for token in tokens:
        if token in _token_dict:
            token_dict[token] = len(token_dict)
            keep_tokens.append(_token_dict[token])
    # 使用新词典构建分词器
    tokenizer = Tokenizer(token_dict, do_lower_case=True)
    return tokenizer, keep_tokens


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


class MyDataGenerator(DataGenerator):

    def __init__(self, data, tokenizer, batch_size=32):
        super(MyDataGenerator, self).__init__(data, batch_size=batch_size)
        self.tokenizer = tokenizer

    def __iter__(self, random=False):
        # 混洗数据
        if random:
            np.random.shuffle(self.data)
        # 迭代整个数据集，每次返回一个mini batch
        total = len(self.data)
        for start in range(0, total, self.batch_size):
            end = min(start + self.batch_size, total)
            batch_token_ids = []
            batch_segment_ids = []
            batch_labels = []
            # 逐样本进行处理
            for _sentence, _label in self.data[start:end]:
                token_ids, segment_ids = self.tokenizer.encode(_sentence)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append(_label)
            # 填充到相同长度
            batch_token_ids = sequence_padding(batch_token_ids, length=settings.MAX_LEN)
            batch_segment_ids = sequence_padding(batch_segment_ids, length=settings.MAX_LEN)
            batch_labels = np.reshape(np.array(batch_labels), (-1, 1))
            yield [batch_token_ids, batch_segment_ids], batch_labels
            del batch_token_ids
            del batch_segment_ids
            del batch_labels


# 加载数据集
sentences, labels = load_data()
# 精简词表，创建tokenizer
tokenizer, keep_tokens = create_tokenizer(sentences)
# 划分数据集
train_data, dev_data, test_data = split_dataset(sentences, labels)
