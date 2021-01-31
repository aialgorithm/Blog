"""
预测DAU指标:
ARIMA: 差分自回归移动平均模型


"""


import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
from math import sqrt
from pandas import Series
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import acf, pacf,plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA, ARMA, ARIMAResults
from statsmodels.tsa.stattools import adfuller as ADF
from keras.models import load_model


warnings.filterwarnings("ignore")

# tz=0 DAU_sum
df = pd.read_csv('./data/DAU_tz0.csv')
dau = df["value"]
print(df.head(10))

# 折线图
df.plot()
plt.show()

# 箱线图
ax = sns.boxplot(y=dau)
plt.show()


# pearsonr时间相关性
# a = df['dau']
# b = df.index
# print(scipy.stats.pearsonr(a,b))
# #-0.3262027101414732, 0.006632465474719386
# 自相关性
plot_acf(dau)
plot_pacf(dau)
plt.show()
print('raw序列的ADF')
# p值大于0.05为非平衡时间序列
print(ADF(dau))
#(-0.4414083594376254,P值 0.9029521269146843, 0, 67,
# {'1%': -3.5319549603840894, '5%': -2.905755128523123, '10%': -2.5903569458676765}, 1330.0486916403536)


# dau_log = np.log(dau)

# dau_log = dau_log.ewm(com=0.5, span=12).mean()
# plot_acf(dau_log)
# plot_pacf(dau_log)
# plt.show()
# print('log序列的ADF')
# print(ADF(dau_log))
# #(0.04817644153532021, 0.9623394783259831, 2, 65,
# # {'1%': -3.5352168748293127, '5%': -2.9071540828402367, '10%': -2.5911025443786984}, -3.563662268845718)
# print('log序列的白噪声检验结果')
# # 大于0.05为白噪声序列
# print(acorr_ljungbox(dau_log, lags=1))
# #(array([50.3883441]), array([1.26141206e-12]))

#差分平稳处理
diff_1_df = dau.diff(1).dropna(how=any)
diff_1_df = diff_1_df
diff_1_df.plot()
plot_acf(diff_1_df)
plot_pacf(diff_1_df)
plt.show()

print('差分序列的ADF')
print(ADF(diff_1_df))
#(-7.296820308000623, p1 1.3710560053434156e-10, 0, 66,
# {'1%': -3.5335601309235605, '5%': -2.9064436883991434, '10%': -2.590723948576676}, 1306.8499722912552)
# (-0.913770472695999, 0.7834037847008933, 4, 61,
# {'1%': -3.542412746661615, '5%': -2.910236235808284, '10%': -2.5927445767266866}, 1294.1719644274262)
print('差分序列的白噪声检验结果')
# 大于0.05为白噪声序列
print(acorr_ljungbox(diff_1_df, lags=1))
# (array([0.69570612]), P1值array([0.40423027]))
#(array([18.06943954]), P2array([2.12992775e-05]))
#(array([5.69722348]),  logp1 array([0.01699177]))

# # 给出最优p q值 ()
r, rac, Q = sm.tsa.acf(diff_1_df, qstat=True)
prac = pacf(diff_1_df, method='ywmle')
table_data = np.c_[range(1,len(r)), r[1:], rac, prac[1:len(rac)+1], Q]
table = pd.DataFrame(table_data, columns=['lag', "AC","Q", "PAC", "Prob(>Q)"])
order = sm.tsa.arma_order_select_ic(diff_1_df, max_ar=7, max_ma=7, ic=['aic', 'bic', 'hqic'])
p, q =order.bic_min_order
print("p,q")
print(p, q)

# 建立ARIMA(0, 1, 1)模型
order = (p, 1, q)
train_X = diff_1_df[:]
arima_model = ARIMA(train_X, order).fit()

# 模型报告
# print(arima_model.summary2())

# 保存模型
arima_model.save('./data/arima_model.h5')

# # load model
arima_model = ARIMAResults.load('./data/arima_model.h5')


# 预测未来两天数据
predict_data_02 = arima_model.predict(start=len(train_X), end=len(train_X) + 1, dynamic = False)

# 预测历史数据
predict_data = arima_model.predict(dynamic = False)

# 逆log差分
# original_series = np.exp(train_X.values[1:] + np.log(dau.values[1:-1]))
# predict_series = np.exp(predict_data.values + np.log(dau.values[1:-1]))
# 逆差分
original_series = train_X.values[1:] + dau.values[1:-1]
predict_series = predict_data.values + dau.values[1:-1]

# comp = pd.DataFrame()
# comp['original'] = original_series
# comp['predict'] = predict_series
split_num = int(len(dau.values)/3) or 1
rmse = sqrt(mean_squared_error(original_series[-split_num:], predict_series[-split_num:]))
print('Test RMSE: %.3f' % rmse)
# (0,1,0)Test RMSE

plt.title('ARIMA RMSE: %.3f' % rmse)
plt.plot(original_series[-split_num:], label="original_series")
plt.plot(predict_series[-split_num:], label="predict_series")
plt.legend()
plt.show()