"""
Author: 公众号-算法进阶
基于启发式双向搜索及模拟退火的特征选择方法。
"""


import pandas as pd 
import random 

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc


def model_metrics(model, x, y, pos_label=1):
    """ 
    评价函数 
    
    """
    yhat = model.predict(x)
    yprob = model.predict_proba(x)[:,1]
    fpr, tpr, _ = roc_curve(y, yprob, pos_label=pos_label)
    result = {'accuracy_score':accuracy_score(y, yhat),
              'f1_score_macro': f1_score(y, yhat, average = "macro"),
              'precision':precision_score(y, yhat,average="macro"),
              'recall':recall_score(y, yhat,average="macro"),
              'auc':auc(fpr,tpr),
              'ks': max(abs(tpr-fpr))
             }
    return result

def bidirectional_selection(model, x_train, y_train, x_test, y_test, annealing=True, anneal_rate=0.2, iters=10,best_metrics=0,
                         metrics='auc',threshold_in=0.0001, threshold_out=0.0001,early_stop=True, 
                         verbose=True):
    """
    model  选择的模型
    annealing     模拟退火算法
    anneal_rate   退火概率，随迭代采纳概率衰减
    threshold_in  特征入模的>阈值
    threshold_out 特征剔除的<=阈值
    """
    included = []
    best_metrics = best_metrics
    
    for i in range(iters):
        # forward step     
        print("iters", i)
        changed = False 
        excluded = list(set(x_train.columns) - set(included))
        random.shuffle(excluded) 
        for new_column in excluded:             
            model.fit(x_train[included+[new_column]], y_train)
            latest_metrics = model_metrics(model, x_test[included+[new_column]], y_test)[metrics]
            if latest_metrics - best_metrics > threshold_in:
                included.append(new_column)
                change = True 
                if verbose:
                    print ('Add {} with metrics gain {:.6}'.format(new_column,latest_metrics-best_metrics))
                best_metrics = latest_metrics
            elif annealing:
                if random.randint(0, i) / iters <= anneal_rate:
                    included.append(new_column)
                    if verbose:
                        print ('Annealing Add   {} with metrics gain {:.6}'.format(new_column,latest_metrics-best_metrics))
                    
        # backward step                      
        random.shuffle(included)
        for new_column in included:
            included.remove(new_column)
            model.fit(x_train[included], y_train)
            latest_metrics = model_metrics(model, x_test[included], y_test)[metrics]
            if latest_metrics - best_metrics <= threshold_out:
                included.append(new_column)
            else:
                changed = True 
                best_metrics= latest_metrics 
                if verbose:
                    print('Drop{} with metrics gain {:.6}'.format(new_column,latest_metrics-best_metrics))
        if not changed and early_stop:
            break 
    return included      

#示例
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y)

model = LGBMClassifier()
included =  bidirectional_selection(model, x_train, y_train, x_test, y_test, annealing=True, iters=50,best_metrics=0.5,
                     metrics='auc',threshold_in=0.0001, threshold_out=0,
                     early_stop=False,verbose=True)