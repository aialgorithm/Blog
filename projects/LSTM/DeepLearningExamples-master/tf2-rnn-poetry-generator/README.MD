# tf2-rnn-poetry-generator

这是一个使用TensorFlow 2.0编写的、基于循环神经网络实现的一个古诗生成器，主要包括如下功能：

- 使用唐诗数据集训练模型。
- 使用训练好的模型，随机生成一首古体诗。
- 使用训练好的模型，续写一首古体诗。
- 使用训练好的模型，随机生成一首藏头诗。

随机生成一首古体诗：

```
金鹤有僧心，临天寄旧身。
石松惊枕树，红鸟发禅新。
不到风前远，何人怨夕时。
明期多尔处，闲此不依迟。
```

```
水泉临鸟声，北去暮空行。
林阁多开雪，楼庭起洞城。
夜来疏竹外，柳鸟暗苔清。
寂寂重阳里，悠悠一钓矶。
```

续写一首古体诗（以"床前明月光，"为例）：

```
床前明月光，翠席覆银丝。
岁气分龙阁，无人入鸟稀。
圣明无泛物，云庙逐雕旗。
永夜重江望，南风正送君。
```

```
床前明月光，清水入寒云。
远景千山雨，萧花入翠微。
影云虚雪润，花影落云斜。
独去江飞夜，谁能作一花。
```

随机生成一首藏头诗（以"海阔天空"为例）：

```
海口多无定，
阔庭何所难。
天山秋色上，
空石昼尘连。
```

```
海庭愁不定，
阔处到南关。
天阙青秋上，
空城雁渐催。
```

# 快速开始

## 1. 如何训练模型？

- clone项目到本地。
- 安装依赖环境。
- 在`settings.py`中配置好相关路径。
- 在`tf2-rnn-poetry-generator`路径下，执行命令`python3 train.py`。

## 2. 如何使用训练好的模型生成古诗？

- clone项目到本地。
- 安装依赖环境。
- 在`settings.py`中配置好相关路径。
- 在`tf2-rnn-poetry-generator`路径下，执行命令`python3 eval.py`。

# 错误排除

1.当使用PyCharm打开并调试项目时，发现py文件的引用路径有误，怎么办？

答：参考[错误排除](https://github.com/AaronJny/DeepLearningSamples/tree/master/tf2-rnn-poetry-generator)。

# 更多信息

请访问博客：

[有趣的深度学习——使用TensorFlow 2.0 + RNN 实现一个古体诗生成器](https://www.aaronjny.com/articles/2020/01/02/1577944439534.html) (https://www.aaronjny.com/articles/2020/01/02/1577944439534.html)

或

[有趣的深度学习——使用TensorFlow 2.0 + RNN 实现一个古体诗生成器](https://blog.csdn.net/aaronjny/article/details/103806954) (https://blog.csdn.net/aaronjny/article/details/103806954)
