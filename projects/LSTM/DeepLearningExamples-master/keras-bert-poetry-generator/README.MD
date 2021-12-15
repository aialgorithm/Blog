# keras-bert-poetry-generator

这是一个基于Google开源的BERT模型编写的一个古诗生成器，主要包括如下功能：

- 使用唐诗数据集训练模型。
- 使用训练好的模型，随机生成一首古体诗。
- 使用训练好的模型，续写一首古体诗。
- 使用训练好的模型，随机生成一首藏头诗。

随机生成一首古体诗：

```
不见山头寺，唯闻竹下僧。
白云生寺远，青壁入山深。
夜宿高楼月，秋眠白阁钟。
不缘山下路，何事见僧踪。
```

```
千里彩云千里别，再来还访玉京师。
三年不负青云志，此地终须见汝时。
```

续写一首古体诗（以"床前明月光，"为例）：

```
床前明月光，无端出曙寒。
夜来应不寐，头白露沾袍。
```

```
床前明月光，不见到天涯。
寂寞海云外，寥寥孤烛前。
```

随机生成一首藏头诗（以"海阔天空"为例）：

```
海燕朝朝去，
阔鸥还远居。
天寒疑水势，
空见见鱼行。
```

```
海上苍须插锦鳞，
阔无心计似文君。
天涯本是无心物，
空解将人学钓鳌。
```

BERT模型使用的是[苏剑林](https://github.com/bojone)大佬封装的[bert4keras](https://github.com/bojone/bert4keras)，开发过程中参考了bert4keras的[demo](https://github.com/bojone/bert4keras/tree/master/examples)。

模型参数使用的是Google官方在中文语料上的预训练权重[BERT-Base, Chinese](https://github.com/google-research/bert#pre-trained-models)，下载地址为[https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

# 快速开始

## 1. 如何训练模型？

- clone项目到本地。
- 下载Google的预训练权重。
- 安装依赖环境。
- 在`settings.py`中配置好相关路径。
- 在`keras-bert-poetry-generator`路径下，执行命令`python3 train.py`。

## 2. 如何使用训练好的模型生成古诗？

- clone项目到本地。
- 下载Google的预训练权重。
- 安装依赖环境。
- 在`settings.py`中配置好相关路径。
- 在`keras-bert-poetry-generator`路径下，执行命令`python3 eval.py`。

# 错误排除

1.当使用PyCharm打开并调试项目时，发现py文件的引用路径有误，怎么办？

答：参考[错误排除](https://github.com/AaronJny/DeepLearningSamples/tree/master/tf2-rnn-poetry-generator)。