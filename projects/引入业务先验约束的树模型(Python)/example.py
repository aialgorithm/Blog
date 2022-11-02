import os
import shutil
import logging
import argparse
import pandas as pd
import os
 
os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz\\bin'
from GBDT.gbdt import GradientBoostingRegressor
from GBDT.gbdt import GradientBoostingBinaryClassifier
from GBDT.gbdt import GradientBoostingMultiClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.removeHandler(logger.handlers[0])
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)


def get_data(model):
    dic = {}
    dic['regression'] = [pd.DataFrame(data=[[1, 5, 20, 1.1],
                                  [2, 7, 30, 1.3],
                                  [3, 21, 70, 1.7],
                                  [4, 30, 60, 1.8],
                                  ], columns=['id', 'age', 'weight', 'label']),
                         pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]

    dic['binary_cf'] = [pd.DataFrame(data=[[1, 5, 20, 0],
                        [2, 7, 30, 0],
                        [3, 21, 70, 1],
                        [4, 30, 60, 1],
                        ], columns=['id', 'age', 'weight', 'label']),
                        pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]

    dic['multi_cf'] = [pd.DataFrame(data=[[1, 5, 20, 0],
                        [2, 7, 30, 0],
                        [3, 21, 70, 1],
                        [4, 30, 60, 1],
                        [5, 30, 60, 2],
                        [6, 30, 70, 2],
                        ], columns=['id', 'age', 'weight', 'label']),
                       pd.DataFrame(data=[[5, 25, 65]], columns=['id', 'age', 'weight'])]

    return dic[model]


def run(args):
    """
    feas_logit 配置特征业务逻辑性得约束, 如新增{'age': -1}，
    表示从个人业务理解上，年龄age应该和标签是呈现负相关的，年龄数值越大，标签值越小；
    """
    ### 配置特征业务逻辑性得约束###
    feas_logit = {'age': -1}
    ### 配置end###
    model = None
    # 获取训练和测试数据
    data = get_data(args.model)[0]
    test_data = get_data(args.model)[1]
    # 创建模型结果的目录
    if not os.path.exists('results'):
        os.makedirs('results')
    if len(os.listdir('results')) > 0:
        shutil.rmtree('results')
        os.makedirs('results')
    # 初始化模型
    if args.model == 'regression':
        model = GradientBoostingRegressor(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
                                          min_samples_split=args.count, is_log=args.log, is_plot=args.plot)
    elif args.model == 'binary_cf':
        model = GradientBoostingBinaryClassifier(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth,
                                                 is_log=args.log, is_plot=args.plot,feas_logit=feas_logit)
    elif args.model == 'multi_cf':
        model = GradientBoostingMultiClassifier(learning_rate=args.lr, n_trees=args.trees, max_depth=args.depth, is_log=args.log,is_plot=args.plot)
    # 训练模型
    model.fit(data)
    # 记录日志
    logger.removeHandler(logger.handlers[-1])
    logger.addHandler(logging.FileHandler('results/result.log'.format(iter), mode='w', encoding='utf-8'))
    logger.info(data)
    # 模型预测
    model.predict(test_data)
    # 记录日志
    logger.setLevel(logging.INFO)
    if args.model == 'regression':
        logger.info((test_data['predict_value']))
    if args.model == 'binary_cf':
        logger.info((test_data['predict_proba']))
        logger.info((test_data['predict_label']))
    if args.model == 'multi_cf':
        logger.info((test_data['predict_label']))
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GBDT-Simple-Tutorial')
    parser.add_argument('--model', default='regression', help='the model you want to use',
                        choices=['regression', 'binary_cf', 'multi_cf'])
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--trees', default=5, type=int, help='the number of decision trees')
    parser.add_argument('--depth', default=3, type=int, help='the max depth of decision trees')
    # 非叶节点的最小数据数目，如果一个节点只有一个数据，那么该节点就是一个叶子节点，停止往下划分
    parser.add_argument('--count', default=2, type=int, help='the min data count of a node')
    parser.add_argument('--log', default=False, type=bool, help='whether to print the log on the console')
    parser.add_argument('--plot', default=True, type=bool, help='whether to plot the decision trees')
    args = parser.parse_args()
    run(args)
    pass
