
import numpy as np
from pandas import DataFrame, datetime, concat,read_csv, Series
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller as ADF

import xgboost as xgb
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import GridSearchCV


from sklearn.externals import joblib


from sklearn.metrics import mean_squared_error, r2_score


# convert date
def parser(x):
    return datetime.strptime(x,"%Y-%m-%d")


#supervised
def timeseries_to_supervised(data, lag=1):
  df = DataFrame(data)
  columns = [df.shift(i) for i in range(1, lag+1)]
  columns.append(df)
  df = concat(columns, axis=1)
  df.fillna(0, inplace=True)
  return df

# diff series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert diff value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# 数据max_min标准化
# scale train and test data to [-1, 1]
def scale(train, test):
  # fit scaler
  scaler = MinMaxScaler(feature_range=(-1, 1))
  scaler = scaler.fit(train)
  # transform train
  train = train.reshape(train.shape[0], train.shape[1])
  train_scaled = scaler.transform(train)
  # transform test2D
  test = test.reshape(test.shape[0], test.shape[1])
  test_scaled = scaler.transform(test)
  return scaler, train_scaled, test_scaled

# invert scale transform
def invert_scale(scaler, X, value):
  new_row = [x for x in X] + [value]
  array = np.array(new_row)
  array = array.reshape(1, len(array))
  inverted = scaler.inverse_transform(array)
  return inverted[0, -1]

# model train
def fit_xgb(train):
  xgb_model = xgb.XGBRegressor(seed=1)
  X, y = train[:,0:-1], train[:,-1]
  xgb_model.fit(X, y)
  return xgb_model

# load data
series = read_csv('./data/DAU_tz0.csv')["value"]
print(series.head())
series.plot()
plt.show()

# 数据平稳化
raw_values = series.values
diff_values = difference(raw_values, 1)
print(diff_values.head())
plt.plot(raw_values, label="raw")
plt.plot(diff_values, label="diff")
plt.legend()
plt.show()
print('差分序列的ADF')
print(ADF(diff_values)[1])
print('差分序列的白噪声检验结果')
# 大于0.05为白噪声序列
#差分序列的ADF
# (-5.626262330867963, 1.1155939417390791e-06, 13, 512, {'1%': -3.443186695642769, '5%': -2.86720156693697, '10%': -2.569785402984619}, 11125.656632822487)
# 差分序列的白噪声检验结果
# (array([13.95689179]), array([0.00018705]))
print(acorr_ljungbox(diff_values, lags=1)[1][0])
# 序列转监督数据
supervised = timeseries_to_supervised(diff_values, 1)
print(supervised.head())
supervised_values = supervised.values

# 训练集测试集
split_num = int(len(supervised_values)/3) or 1
train, test = supervised_values[0:-split_num], supervised_values[-split_num:]

# 标准化
scaler, train_scaled, test_scaled = scale(train, test)

#模型训练
xgb_model = fit_xgb(train_scaled)

# 保存模型
joblib.dump(xgb_model, "./data/xgb_model")

# # load model
xgb_model = joblib.load("./data/xgb_model")

# validation

# test_scaled_x = test_scaled[:,0:-1]
#
# predictions_scaled = xgb_model.predict(test_scaled_x)
## pandas
# invert scaling
# predictions_scaled = predictions_scaled.reshape(len(predictions_scaled), 1)
# test_scaled_0 = [DataFrame(test_scaled[:,0])]
# test_scaled_0.append(DataFrame(predictions_scaled))
#
# con_predictions_scaled = concat(test_scaled_0,axis=1)
#
# inverted = scaler.inverse_transform(con_predictions_scaled)

# yhat = inverted[:,-1]
# validation
predictions = list()
for i in range(len(test_scaled)):
  # make one-step forecast
  X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
  yhat = xgb_model.predict([X])
  # invert scaling
  yhat = invert_scale(scaler, X, yhat)
  # invert differencing
  yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
  # store forecast
  predictions.append(yhat)
  expected = raw_values[len(train) + i + 1]
  mae = abs(yhat-expected)
  print('data=%d, Predicted=%f, Expected=%f, mae=%.3f' % (i+1, yhat, expected, mae))
mae = np.average(abs(predictions - raw_values[-split_num:]))
print("Test MAE: %.3f",mae)
#report performance
rmse = sqrt(mean_squared_error(raw_values[-split_num:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
plt.plot(raw_values[-split_num:], label="raw")
plt.plot(predictions, label="predict")
plt.title('XGBoost Test RMSE: %.3f' % rmse)
plt.legend()
plt.show()
