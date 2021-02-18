#!/usr/bin/env python
# coding: utf-8

# ### 这是个简单demo：使用iris植物的数据，训练iris分类模型，通过模型预测识别iris品种


# 导入模块
import pandas as pd
from sklearn.datasets import load_iris

# 加载数据集 
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['class'] = data.target
df.head()

# pandas_profiling是一个超实用的数据分析模块，使用它可快速数据缺失情况、数据分布、相关情况
import pandas_profiling

df.profile_report(title='iris')

# 特征工程 
# （略）该数据集质量较高，不可以不用数据清洗，缺失值填充等

# 划分目标标签y、特征x
y = df['class']
x = df.drop('class', axis=1)

#划分训练集，测试集
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(x, y)


# 模型训练
from xgboost import XGBClassifier

# 选择XGBoost模型 及 调试参数
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=1,
              min_child_weight=1, missing=None, n_estimators=1, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

# 训练模型
xgb.fit(train_x, train_y)

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc


def model_metrics(model, x, y, pos_label=2):
    """
    评估函数
    """
    yhat = model.predict(x)
    result = {'accuracy_score':accuracy_score(y, yhat),
              'f1_score_macro': f1_score(y, yhat, average = "macro"),
              'precision':precision_score(y, yhat,average="macro"),
              'recall':recall_score(y, yhat,average="macro")
             }
    return result


# 模型评估结果
print("TRAIN")
print(model_metrics(xgb, train_x, train_y))

print("TEST")
print(model_metrics(xgb, test_x, test_y))


# 模型预测
xgb.predict(test_x)