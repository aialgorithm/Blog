# tf2-rnn-emotional-classifier

这是一个基于 TensorFlow2 + RNN 的情感分析Demo,数据集使用的是新浪微博的情感标注数据集，共约10万条，其中正负向评论各约5万条。

数据集来自于 [InsaneLife/ChineseNLPCorpus](https://github.com/InsaneLife/ChineseNLPCorpus#%E6%83%85%E6%84%9F%E8%A7%82%E7%82%B9%E8%AF%84%E8%AE%BA-%E5%80%BE%E5%90%91%E6%80%A7%E5%88%86%E6%9E%90) 的收集。

我的计算设备是笔记本上的 GTX 1060 6G，当选择batch size=32时，训练完一个epoch耗时约125s。

如下是训练4个epoch的模型性能参数：


数据集|准确率(Accuracy)|精确率(Precision)|召回率(Recall)|F1 Score
--|--|--|--|--
训练集|0.9759|0.9904|0.9610|0.9746
开发集(验证集)|0.9732|0.9858|0.9605|0.9722
测试集|0.9753|0.9866|0.9634|0.9740


数据集划分比例为：

数据集|比例
--|--
训练集|0.75
开发集|0.15
验证集|0.15


# 快速开始

我的Python环境是Linux下使用Anaconda安装的，所以请根据自己的实际情况，自行判断是否需要将下面命令中的python和pip替换成python3和pip3。

## 1.如何训练模型？

- clone项目到本地。
- 下载数据集。
- 在`tf2-rnn-emotional-classifier`路径下，执行命令`pip install -r requirements.txt`
- 在`settings.py`中配置好相关路径，并根据个人情况配置好其他参数。
- 在`tf2-rnn-emotional-classifier`路径下，执行命令`python train.py`。