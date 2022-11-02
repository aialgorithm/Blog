"""
Created on ：2019/03/28
@author: Freeman, feverfc1994
"""

import abc
import math
import logging
import pandas as pd
from GBDT.decision_tree import Tree
from GBDT.loss_function import SquaresError, BinomialDeviance, MultinomialDeviance
from GBDT.tree_plot import plot_tree, plot_all_trees,plot_multi
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def predict(self, data):
        pass


class BaseGradientBoosting(AbstractBaseGradientBoosting):

    def __init__(self, loss, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False,feas_logit={}):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = None
        self.trees = {}
        self.f_0 = {}
        self.is_log = is_log
        self.is_plot = is_plot
        self.feas_logit = feas_logit

    def fit(self, data):
        """
        :param data: pandas.DataFrame, the features data of train training   
        """
        # 掐头去尾， 删除id和label，得到特征名称
        self.features = list(data.columns)[1: -1]
        # 初始化 f_0(x)
        # 对于平方损失来说，初始化 f_0(x) 就是 y 的均值
        self.f_0 = self.loss.initialize_f_0(data)
        # 对 m = 1, 2, ..., M
        logger.handlers[0].setLevel(logging.INFO if self.is_log else logging.CRITICAL)
        for iter in range(1, self.n_trees+1):
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
            fh = logging.FileHandler('results/NO.{}_tree.log'.format(iter), mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            # 计算负梯度--对于平方误差来说就是残差
            logger.info(('-----------------------------构建第%d颗树-----------------------------' % iter))
            self.loss.calculate_residual(data, iter)
            target_name = 'res_' + str(iter)
            self.trees[iter] = Tree(data, self.feas_logit,self.max_depth, self.min_samples_split,
                                    self.features, self.loss, target_name, logger)
            self.loss.update_f_m(data, self.trees, iter, self.learning_rate, logger)
            if self.is_plot:
                plot_tree(self.trees[iter], max_depth=self.max_depth, iter=iter)
        # print(self.trees)
        if self.is_plot:
            plot_all_trees(self.n_trees)


class GradientBoostingRegressor(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False,feas_logit={}):
        super().__init__(SquaresError(), learning_rate, n_trees, max_depth,
                         min_samples_split, is_log, is_plot,feas_logit)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees+1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_value'] = data[f_m_name]


class GradientBoostingBinaryClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False,feas_logit={}):
        super().__init__(BinomialDeviance(), learning_rate, n_trees, max_depth,
                         min_samples_split, is_log, is_plot,feas_logit)

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees + 1):
            f_prev_name = 'f_' + str(iter - 1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        data['predict_proba'] = data[f_m_name].apply(lambda x: 1 / (1 + math.exp(-x)))
        data['predict_label'] = data['predict_proba'].apply(lambda x: 1 if x >= 0.5 else 0)


class GradientBoostingMultiClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__(MultinomialDeviance(), learning_rate, n_trees, max_depth,
                         min_samples_split, is_log, is_plot)

    def fit(self, data):
        # 掐头去尾， 删除id和label，得到特征名称
        self.features = list(data.columns)[1: -1]
        # 获取所有类别
        self.classes = data['label'].unique().astype(str)
        # 初始化多分类损失函数的参数 K
        self.loss.init_classes(self.classes)
        # 根据类别将‘label’列进行one-hot处理
        for class_name in self.classes:
            label_name = 'label_' + class_name
            data[label_name] = data['label'].apply(lambda x: 1 if str(x) == class_name else 0)
            # 初始化 f_0(x)
            self.f_0[class_name] = self.loss.initialize_f_0(data, class_name)
        # print(data)
        # 对 m = 1, 2, ..., M
        logger.handlers[0].setLevel(logging.INFO if self.is_log else logging.CRITICAL)
        for iter in range(1, self.n_trees + 1):
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
            fh = logging.FileHandler('results/NO.{}_tree.log'.format(iter), mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            logger.info(('-----------------------------构建第%d颗树-----------------------------' % iter))
            # 这里计算负梯度整体计算是为了计算p_sum的一致性
            self.loss.calculate_residual(data, iter)
            self.trees[iter] = {}
            for class_name in self.classes:
                target_name = 'res_' + class_name + '_' + str(iter)
                self.trees[iter][class_name] = Tree(data, self.feas_logit, self.max_depth, self.min_samples_split,
                                                    self.features, self.loss, target_name, logger)
                self.loss.update_f_m(data, self.trees, iter, class_name, self.learning_rate, logger)
            if self.is_plot:
                    plot_multi(self.trees[iter], max_depth=self.max_depth, iter=iter)
        if self.is_plot:
            plot_all_trees(self.n_trees)

    def predict(self, data):
        """
        此处的预测的实现方式和生成树的方式不同，
        生成树是需要每个类别的树的每次迭代需要一起进行，外层循环是iter，内层循环是class
        但是，预测时树已经生成，可以将class这层循环作为外循环，可以节省计算成本。
        """
        for class_name in self.classes:
            f_0_name = 'f_' + class_name + '_0'
            data[f_0_name] = self.f_0[class_name]
            for iter in range(1, self.n_trees + 1):
                f_prev_name = 'f_' + class_name + '_' + str(iter - 1)
                f_m_name = 'f_' + class_name + '_' + str(iter)
                data[f_m_name] = \
                    data[f_prev_name] + \
                    self.learning_rate * data.apply(lambda x:
                                                    self.trees[iter][class_name].root_node.get_predict_value(x), axis=1)

        data['sum_exp'] = data.apply(lambda x:
                                     sum([math.exp(x['f_' + i + '_' + str(iter)]) for i in self.classes]), axis=1)

        for class_name in self.classes:
            proba_name = 'predict_proba_' + class_name
            f_m_name = 'f_' + class_name + '_' + str(iter)
            data[proba_name] = data.apply(lambda x: math.exp(x[f_m_name]) / x['sum_exp'], axis=1)
        # TODO: log 每一类的概率
        data['predict_label'] = data.apply(lambda x: self._get_multi_label(x), axis=1)

    def _get_multi_label(self, x):
        label = None
        max_proba = -1
        for class_name in self.classes:
            if x['predict_proba_' + class_name] > max_proba:
                max_proba = x['predict_proba_' + class_name]
                label = class_name
        return label
