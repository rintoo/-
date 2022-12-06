import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

df = pd.read_csv('../de_simulated_data.csv',index_col=0)

# test找参数
# df_train = df.iloc[:199,:]    #['2015-11-23':'2019-09-09']
# df_test = df.iloc[199:,]      #['2019-10-14':'2019-11-11']
# x1 = df_train['competitor_sales_B'].values.reshape(-1,1)
# y1 = df_train['revenue'].values
# x1 = StandardScaler().fit_transform(x1)
# y1 = StandardScaler().fit_transform(y1.reshape(-1,1)).reshape(-1)
# regressor = SVR(kernel = 'rbf',gamma=0.01,C=100)#
# regressor.fit(x1, y1)
# x2_test = df_test['competitor_sales_B'].values.reshape(-1,1)
# y2_test = df_test['revenue'].values
# x2_test = StandardScaler().fit_transform(x2_test)
# y2_test = StandardScaler().fit_transform(y2_test.reshape(-1,1)).reshape(-1)
# y2_hat = regressor.predict(x2_test)
# x2_test_score = regressor.score(x2_test,y2_test)
# print('x2_test_score:',x2_test_score)
# x = df['competitor_sales_B'].values.reshape(-1,1)
# y = df['revenue'].values
# x = StandardScaler().fit_transform(x)
# y = StandardScaler().fit_transform(y.reshape(-1,1)).reshape(-1)
# y_hat = regressor.predict(x)
# x_score = regressor.score(x,y)
# print('x_score:',x_score)
# plt.subplot(1,2,1)
# plt.plot(x,y,'r.',label='real')
# plt.plot(x,y_hat,'go',label='SVR_simulate')
# plt.ylabel("revenue")
# plt.xlabel("competitor_sales_B")
# plt.title("全部时间上的拟合情况")
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(x2_test,y2_test,'r.',label='real')
# plt.plot(x2_test,y2_hat,'go',label='SVR_simulate')
# plt.ylabel("revenue")
# plt.xlabel("competitor_sales_B")
# plt.title("'2019-10-14'至'2019-11-11'的预测情况")
# plt.legend()
# plt.show()
# print()



# 花费渠道
# label_m = ['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S']
# num_m = len(label_m)
# df_train = df.iloc[:199,:]    #['2015-11-23':'2019-09-09']
# df_test = df.iloc[199:,]      #['2019-10-14':'2019-11-11']
# x1 = df_train[label_m].values.reshape(-1,num_m)
# y1 = df_train['revenue'].values
# x1 = StandardScaler().fit_transform(x1)
# y1 = StandardScaler().fit_transform(y1.reshape(-1,1)).reshape(-1)
# regressor = SVR(kernel = 'rbf',gamma=0.01,C=100)#
# regressor.fit(x1, y1)
# x2_test = df_test[label_m].values.reshape(-1,num_m)
# y2_test = df_test['revenue'].values
# x2_test = StandardScaler().fit_transform(x2_test)
# y2_test = StandardScaler().fit_transform(y2_test.reshape(-1,1)).reshape(-1)
# y2_hat = regressor.predict(x2_test)
# x2_test_score = regressor.score(x2_test,y2_test)
# print('x2_test_score:',x2_test_score)
# x = df[label_m].values.reshape(-1,num_m)
# y = df['revenue'].values
# x = StandardScaler().fit_transform(x)
# y = StandardScaler().fit_transform(y.reshape(-1,1)).reshape(-1)
# y_hat = regressor.predict(x)
# x_score = regressor.score(x,y)
# print('x_score:',x_score)
# plt.subplot(2,1,1)
# plt.plot(y,'r-',label='real')
# plt.plot(y_hat,'g--',label='SVR_simulate')
# plt.ylabel("revenue")
# plt.xlabel("时间")
# plt.title("全部时间上的拟合情况")
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(y2_test,'r-',label='real')
# plt.plot(y2_hat,'g--',label='SVR_simulate')
# plt.ylabel("revenue")
# plt.xlabel("时间")
# plt.title("'2019-10-14'至'2019-11-11'的预测情况")
# plt.legend()
# plt.tight_layout()
# plt.show()
# print()

# 相关变量+花费渠道
label_m = ['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S','facebook_I','search_clicks_P','competitor_sales_B']
num_m = len(label_m)
df_train = df.iloc[:199,:]    #['2015-11-23':'2019-09-09']
df_test = df.iloc[199:,]      #['2019-10-14':'2019-11-11']
x1 = df_train[label_m].values.reshape(-1,num_m)
y1 = df_train['revenue'].values
x1 = StandardScaler().fit_transform(x1)
y1 = StandardScaler().fit_transform(y1.reshape(-1,1)).reshape(-1)
regressor = SVR(kernel = 'rbf',gamma=0.01,C=100)#
regressor.fit(x1, y1)
x2_test = df_test[label_m].values.reshape(-1,num_m)
y2_test = df_test['revenue'].values
x2_test = StandardScaler().fit_transform(x2_test)
y2_test = StandardScaler().fit_transform(y2_test.reshape(-1,1)).reshape(-1)
y2_hat = regressor.predict(x2_test)
x2_test_score = regressor.score(x2_test,y2_test)
print('x2_test_score:',x2_test_score)
x = df[label_m].values.reshape(-1,num_m)
y = df['revenue'].values
x = StandardScaler().fit_transform(x)
y = StandardScaler().fit_transform(y.reshape(-1,1)).reshape(-1)
y_hat = regressor.predict(x)
x_score = regressor.score(x,y)
print('x_score:',x_score)
plt.subplot(2,1,1)
plt.plot(y,'r-',label='real')
plt.plot(y_hat,'g--',label='SVR_simulate')
plt.ylabel("revenue")
plt.xlabel("时间")
plt.title("全部时间上的拟合情况")
plt.legend()
plt.subplot(2,1,2)
plt.plot(y2_test,'r-',label='real')
plt.plot(y2_hat,'g--',label='SVR_simulate')
plt.ylabel("revenue")
plt.xlabel("时间")
plt.title("'2019-10-14'至'2019-11-11'的预测情况")
plt.legend()
plt.tight_layout()
plt.show()