import numpy as np
from pandas import DataFrame, datetime, concat,read_csv, Series
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from keras import regularizers
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller as ADF

# convert date
def parser(x):
    return datetime.strptime(x,"%Y-%m-%d")

#supervised
def timeseries_to_supervised(data, lag=1):
  df = DataFrame(data)
  columns = [df.shift(1) for i in range(1, lag+1)]
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
  # transform test
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
def fit_lstm(train, batch_size, nb_epoch, neurons):
  X, y = train[:,0:-1], train[:,-1]
  # reshp
  X = X.reshape(X.shape[0], 1, X.shape[1])
  model = Sequential()
  # stateful
  # input(samples:batch row, time steps:1, features:one time observed)
  model.add(LSTM(neurons,
                 batch_input_shape=(batch_size, X.shape[1], X.shape[2]),
                 stateful=True, return_sequences=True, dropout=0.2))
  model.add(Dense(1))
  model.compile(loss="mean_squared_error", optimizer="adam")

  # 提前停止
  early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
  # saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,
  #                                            mode='auto')
  # val_loss， loss plot 
  train_loss = []
  val_loss = []
  for i in range(nb_epoch):
    # shuffle=false
    history = model.fit(X, y, batch_size=batch_size, epochs=1,verbose=0,shuffle=False,validation_split=0.3)
    train_loss.append(history.history['loss'][0])
    val_loss.append(history.history['val_loss'][0])
    # clear state
    model.reset_states()
    # 提前停止训练
    if i > 50 and sum(val_loss[-10:]) < 0.3:
      print(sum(val_loss[-5:]))
      print("better epoch", i)
      break

    # print(history.history['loss'])

  plt.plot(train_loss)
  plt.plot(val_loss)
  plt.title('model train vs validation loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper right')
  plt.show()
  return model

# model predict
def forecast_lstm(model, batch_size, X):
  X = X.reshape(1, 1, len(X))
  yhat = model.predict(X, batch_size=batch_size)
  return yhat[0,0]

# 开始加载数据load data
series = read_csv('DAU.csv')["value"]
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
# (array([13.95689179]), array([0.00018705]))
print(acorr_ljungbox(diff_values, lags=1)[1][0])
# 序列转监督数据
supervised = timeseries_to_supervised(diff_values, 1)
print(supervised.head())
supervised_values = supervised.values

# split data
split_num = int(len(supervised_values)/3) or 1
train, test = supervised_values[0:-split_num], supervised_values[-split_num:]

# 标准化
scaler, train_scaled, test_scaled = scale(train, test)

#fit model
lstm_model = fit_lstm(train_scaled, 1, 200, 5)
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
train_predict = lstm_model.predict(train_reshaped, batch_size=1)
train_raw = train_scaled[:, 0]

# # train RMSE plot
# train_raw = raw_values[0:-split_num]
# predictions = list()
# for i in range(len(train_scaled)):
#   # make one-step forecast
#   X, y = train_scaled[i, 0:-1], train_scaled[i, -1]
#   yhat = forecast_lstm(lstm_model, 1, X)
#   # invert scaling
#   yhat = invert_scale(scaler, X, yhat)
#   # invert differencing
#   yhat = inverse_difference(raw_values, yhat, len(train_scaled)+1-i)
#   # store forecast
#   predictions.append(yhat)
#   expected = train_raw[i]
#   mae = abs(yhat-expected)
#   print('data=%d, Predicted=%f, Expected=%f, mae=%.3f' % (i+1, yhat, expected,mae))
#   print(mae)
#
# plt.plot(train_raw, label="train_raw")
# plt.plot(predictions, label="predict")
# plt.legend()
# plt.show()
  

# 保存模型
lstm_model.save('./data/lstm_model_epoch50.h5')

# # load model
lstm_model = load_model('./data/lstm_model_epoch50.h5')

# validation
predictions = list()
for i in range(len(test_scaled)):
  # make one-step forecast
  X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
  yhat = forecast_lstm(lstm_model, 1, X)
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
plt.title('LSTM Test RMSE: %.3f' % rmse)
plt.legend()
plt.show()
