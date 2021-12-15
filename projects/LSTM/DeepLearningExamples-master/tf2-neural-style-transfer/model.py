# -*- coding: utf-8 -*-
# @File    : model.py
# @Author  : AaronJny
# @Time    : 2020/03/13
# @Desc    :
import typing
import tensorflow as tf
import settings


def get_vgg19_model(layers):
    """
    创建并初始化vgg19模型
    :return:
    """
    # 加载imagenet上预训练的vgg19
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    # 提取需要被用到的vgg的层的output
    outputs = [vgg.get_layer(layer).output for layer in layers]
    # 使用outputs创建新的模型
    model = tf.keras.Model([vgg.input, ], outputs)
    # 锁死参数，不进行训练
    model.trainable = False
    return model


class NeuralStyleTransferModel(tf.keras.Model):

    def __init__(self, content_layers: typing.Dict[str, float] = settings.CONTENT_LAYERS,
                 style_layers: typing.Dict[str, float] = settings.STYLE_LAYERS):
        super(NeuralStyleTransferModel, self).__init__()
        # 内容特征层字典 Dict[层名,加权系数]
        self.content_layers = content_layers
        # 风格特征层
        self.style_layers = style_layers
        # 提取需要用到的所有vgg层
        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
        # 创建layer_name到output索引的映射
        self.outputs_index_map = dict(zip(layers, range(len(layers))))
        # 创建并初始化vgg网络
        self.vgg = get_vgg19_model(layers)

    def call(self, inputs, training=None, mask=None):
        """
        前向传播
        :return
            typing.Dict[str,typing.List[outputs,加权系数]]
        """
        outputs = self.vgg(inputs)
        # 分离内容特征层和风格特征层的输出，方便后续计算 typing.List[outputs,加权系数]
        content_outputs = []
        for layer, factor in self.content_layers.items():
            content_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        style_outputs = []
        for layer, factor in self.style_layers.items():
            style_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        # 以字典的形式返回输出
        return {'content': content_outputs, 'style': style_outputs}
