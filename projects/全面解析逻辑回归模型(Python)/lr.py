# coding: utf-8

import numpy as np 
import matplotlib.pyplot as plt
import h5py
import scipy
from sklearn import datasets


# 加载数据并简单划分为训练集/测试集
def load_dataset():
    dataset = datasets.load_breast_cancer()  
    train_x,train_y = dataset['data'][0:400], dataset['target'][0:400]
    test_x, test_y = dataset['data'][400:-1], dataset['target'][400:-1]
    return train_x, train_y, test_x, test_y


# logit激活函数
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))    
    return s
    
# 权重初始化0
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

# 定义学习的目标函数，计算梯度
def propagate(w, b, X, Y):
    m = X.shape[1]      

    A = sigmoid(np.dot(w.T, X) + b)         # 逻辑回归输出预测值  

    cost = -1 / m *  np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))   # 交叉熵损失为目标函数
    dw = 1 / m * np.dot(X, (A - Y).T)   # 计算权重w梯度
    db = 1 / m * np.sum(A - Y)   
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())    
    grads = {"dw": dw,
             "db": db}    
    return grads, cost


# 定义优化算法
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    costs = []    
    for i in range(num_iterations):    # 梯度下降迭代优化
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]              # 权重w梯度
        db = grads["db"]
        w = w - learning_rate * dw   # 按学习率(learning_rate)负梯度(dw)方向更新w
        b = b - learning_rate * db
        if i % 50 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}
    return params, grads, costs


#传入优化后的模型参数w，b，模型预测   
def predict(w, b, X):
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	A = sigmoid(np.dot(w.T, X) + b)
	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_prediction[0, i] = 0
		else:
			Y_prediction[0, i] = 1
	assert(Y_prediction.shape == (1, m))
    
	return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
    # 初始化
    w, b = initialize_with_zeros(X_train.shape[0]) 
    # 梯度下降优化模型参数
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    # 模型预测结果
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    # 模型评估准确率
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}    
    return d
    
    
    
    
    
    
# 加载癌症数据集
train_set_x, train_set_y, test_set_x, test_set_y = load_dataset()   

# reshape
train_set_x = train_set_x.reshape(train_set_x.shape[0], -1).T
test_set_x = test_set_x.reshape(test_set_x.shape[0], -1).T

print(train_set_x.shape)
print(test_set_x.shape)

#训练模型并评估准确率
paras = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 100, learning_rate = 0.001, print_cost = False)

