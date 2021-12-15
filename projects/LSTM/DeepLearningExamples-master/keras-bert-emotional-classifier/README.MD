# keras-bert-emotional-classifier

这是一个基于BERT的情感分析Demo,数据集使用的是新浪微博的情感标注数据集，共约10万条，其中正负向评论各约5万条。

数据集来自于 [InsaneLife/ChineseNLPCorpus](https://github.com/InsaneLife/ChineseNLPCorpus#%E6%83%85%E6%84%9F%E8%A7%82%E7%82%B9%E8%AF%84%E8%AE%BA-%E5%80%BE%E5%90%91%E6%80%A7%E5%88%86%E6%9E%90) 的收集。

我的计算设备是笔记本上的 GTX 1060 6G，算力较低，训练较为耗时，所以只训练了一个epoch(在我的设备上约花费1个小时)，并且取得了还不错的效果。

如下是训练1个epoch的模型性能参数：


数据集|准确率(Accuracy)|精确率(Precision)|召回率(Recall)|F1 Score
--|--|--|--|--
训练集|0.9635|0.9705|0.9565|0.9608
开发集(验证集)|0.9773|0.9913|0.9637|0.9755
测试集|0.9781|0.9915|0.9658|0.9770


数据集划分比例为：

数据集|比例
--|--
训练集|0.75
开发集|0.15
验证集|0.15

模型训练完成后，可以自行输入语句进行情绪倾向判断，如：

```
请输入一句话，模型将判断其情绪倾向：虽然没有买到想要的东西，但我并不沮丧           
正面情绪！
请输入一句话，模型将判断其情绪倾向：没有买到想要的东西， 有点沮丧   
负面情绪！
请输入一句话，模型将判断其情绪倾向：书挺好的，就是贵了点
正面情绪！
请输入一句话，模型将判断其情绪倾向：书的确不错，但也太贵了
负面情绪！
```

BERT模型使用的是[苏剑林](https://github.com/bojone)大佬封装的[bert4keras](https://github.com/bojone/bert4keras)，开发过程中参考了bert4keras的[demo](https://github.com/bojone/bert4keras/tree/master/examples)。

模型参数使用的是Google官方在中文语料上的预训练权重[BERT-Base, Chinese](https://github.com/google-research/bert#pre-trained-models)，下载地址为[https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)


# 快速开始

我的Python环境是Linux下使用Anaconda安装的，所以请根据自己的实际情况，自行判断是否需要将下面命令中的python和pip替换成python3和pip3。

## 1.如何训练模型？

- clone项目到本地。
- 下载Google的预训练权重。
- 安装依赖环境。
- 在`settings.py`中配置好相关路径，并根据个人情况配置好其他参数。
- 在`keras-bert-emotional-classifier`路径下，执行命令`python train.py`。

## 2.如何测试模型？

- clone项目到本地。
- 下载Google的预训练权重。
- 安装依赖环境。
- 在`settings.py`中配置好相关路径。
- 在`keras-bert-emotional-classifier`路径下，执行命令`python eval.py`。
- 根据提示输入句子进行情绪倾向判断。
