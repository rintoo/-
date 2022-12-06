import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


df = pd.read_csv('../de_simulated_data.csv',index_col=0)
y = df['revenue'].values
# y = StandardScaler().fit_transform(y.reshape(-1,1)).reshape(-1)


# # 花费渠道
# label_m = ['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S']
# num_m = len(label_m)
# x = df[label_m].values
# # x = StandardScaler().fit_transform(x)
# df_train = df.iloc[:199,:]    #['2015-11-23':'2019-09-09']
# df_test = df.iloc[199:,]      #['2019-10-14':'2019-11-11']
# x_train = df_train[label_m].values
# # x_train = StandardScaler().fit_transform(x_train)
# y_train = df_train['revenue'].values
# # y_train = StandardScaler().fit_transform(y_train.reshape(-1,1)).reshape(-1)
# x_test = df_test[label_m].values
# # x_test = StandardScaler().fit_transform(x_test)
# y_test = df_test['revenue'].values
# # y_test = StandardScaler().fit_transform(y_test.reshape(-1,1)).reshape(-1)
# # estimator = lgb.LGBMRegressor(num_leaves=31)
# # param_grid = {
# #     'learning_rate': [0.01,0.05,0.07,0.08,0.09, 0.1,0.2,0.3],
# #     'n_estimators': [35, 30,27,26,25,23,22,21]
# # }
# # gbm = GridSearchCV(estimator, param_grid, cv=4)
# # gbm.fit(x_train, y_train)
# # print('Best parameters found by grid search are:', gbm.best_params_) # {'learning_rate': 0.08, 'n_estimators': 25}
# gbm = lgb.LGBMRegressor(objective='regression',num_leaves=31, learning_rate=0.08, n_estimators=25)
# gbm.fit(x_train, y_train, eval_metric='l2')#eval_set=[(x_test, y_test)], , early_stopping_rounds=5
# y_hat = gbm.predict(x)
# print('simulate_all:',gbm.score(x, y))
# y2_hat = gbm.predict(x_test)
# print('predict:',gbm.score(x_test, y_test))
# plt.subplot(2,1,1)
# plt.plot(y,'r-',label='real')
# plt.plot(y_hat,'g--',label='LGBM_simulate')
# plt.ylabel("revenue")
# plt.xlabel("时间")
# plt.title("全部时间上的拟合情况")
# plt.legend()
# plt.subplot(2,1,2)
# plt.plot(y_test,'r-',label='real')
# plt.plot(y2_hat,'g--',label='LGBM_simulate')
# plt.ylabel("revenue")
# plt.xlabel("时间")
# plt.title("'2019-10-14'至'2019-11-11'的预测情况")
# plt.legend()
# plt.tight_layout()
# plt.show()
# print()

# 花费渠道
label_m = ['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S','facebook_I','search_clicks_P','competitor_sales_B']
num_m = len(label_m)
x = df[label_m].values
# x = StandardScaler().fit_transform(x)
df_train = df.iloc[:199,:]    #['2015-11-23':'2019-09-09']
df_test = df.iloc[199:,]      #['2019-10-14':'2019-11-11']
x_train = df_train[label_m].values
# x_train = StandardScaler().fit_transform(x_train)
y_train = df_train['revenue'].values
# y_train = StandardScaler().fit_transform(y_train.reshape(-1,1)).reshape(-1)
x_test = df_test[label_m].values
# x_test = StandardScaler().fit_transform(x_test)
y_test = df_test['revenue'].values
# y_test = StandardScaler().fit_transform(y_test.reshape(-1,1)).reshape(-1)
# estimator = lgb.LGBMRegressor(num_leaves=31)
# param_grid = {
#     'learning_rate': [0.07,0.08,0.09, 0.1,0.2,0.3],
#     'n_estimators': [35, 30,27,26,25,36,37,38,39,40]
# }
# gbm = GridSearchCV(estimator, param_grid, cv=4)
# gbm.fit(x_train, y_train)
# print('Best parameters found by grid search are:', gbm.best_params_) # {'learning_rate': 0.1, 'n_estimators': 38}
gbm = lgb.LGBMRegressor(objective='regression',num_leaves=31, learning_rate=0.1, n_estimators=38)
gbm.fit(x_train, y_train, eval_metric='l2')#eval_set=[(x_test, y_test)], , early_stopping_rounds=5
y_hat = gbm.predict(x)
print('simulate_all:',gbm.score(x, y))
y2_hat = gbm.predict(x_test)
print('predict:',gbm.score(x_test, y_test))
plt.subplot(2,1,1)
plt.plot(y,'r-',label='real')
plt.plot(y_hat,'g--',label='LGBM_simulate')
plt.ylabel("revenue")
plt.xlabel("时间")
plt.title("全部时间上的拟合情况")
plt.legend()
plt.subplot(2,1,2)
plt.plot(y_test,'r-',label='real')
plt.plot(y2_hat,'g--',label='LGBM_simulate')
plt.ylabel("revenue")
plt.xlabel("时间")
plt.title("'2019-10-14'至'2019-11-11'的预测情况")
plt.legend()
plt.tight_layout()
plt.show()
print()