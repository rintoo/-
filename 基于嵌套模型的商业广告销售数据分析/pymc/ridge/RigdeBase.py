import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt

import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge,RidgeCV
'''
P53 http://liu.diva-portal.org/smash/get/diva2:1348365/FULLTEXT01.pdf
Marketing Mix Modelling: A comparative study of statistical models
'''

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题
df = pd.read_csv("de_simulated_data.csv")
# sns.distplot(df['revenue'])
# plt.title('revenue distribution')
# fig = plt.figure()
# res = stats.probplot(df['revenue'], plot=plt)
# plt.show()
# 双峰

def adstockGeometric(x, theta):
    x_decayed = [0 for i in range(len(x))]
    x_decayed[0] = x[0]
    for i in range(len(x)-1):
        x_decayed[i+1] = x[i+1] + theta * x_decayed[i]
    return x_decayed

# def hill(x_normalized, alpha, gamma):
#     #dt = pd.Series(np.array([6, 47, 49, 15, 42, 41, 7, 39, 43, 40, 36])
#     # 直接进行分数线划分
#     gammaTrans = gamma*(max(x_normalized)-min(x_normalized))+min(x_normalized)
#     x_ref = x_normalized**alpha/(x_normalized**alpha+gammaTrans**alpha)
#     return x_ref


df_chn = df[['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S']]
df_chn_VIF = pd.DataFrame()
df_chn_VIF["feature"] = df_chn.columns
# calculating VIF for each feature
df_chn_VIF["VIF"] = [variance_inflation_factor(df_chn.values, i)
                   for i in range(len(df_chn.columns))]

df_chn_VIF.sort_values(by=['VIF'], ascending=False)

# facebook在(0, 0.3)，ooh在(0.1, 0.4)，print在(0.1, 0.4)，tv在(0.3, 0.8)，search在(0, 0.3)

for i in range(3):# facebook
    theta_facebook=i/10

X = ['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S']
dftest = pd.DataFrame()

for x in X:
    for i in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        dftest[x + 'Decay' + str(i)] = adstockGeometric(df_chn[x], i)


#calculate correlation bettwen NP and other variables.
dfmmm_final=pd.concat([df['revenue'],dftest],axis=1)
NPcorr = dfmmm_final.corr()['revenue'][:]
NPcorr = pd.DataFrame(NPcorr).reset_index()
NPcorr.columns=['var','corr']

NPcorr = NPcorr.sort_values(by=['corr'], ascending = False)

dfcorr=pd.DataFrame()
for x in X:
    m = NPcorr[NPcorr['var'].str.contains(x)][:1]
    #m = corr[corr['var'].str.contains(x)][2:3]
    dfcorr = dfcorr.append(m)


# print_SDecay0.4，search_SDecay0.3，facebook_SDecay0.3
dfcorr=pd.DataFrame()
dfcorr['tv_SDecay0.8'] = adstockGeometric(df_chn['tv_S'], 0.8)
dfcorr['ooh_SDecay0.4'] = adstockGeometric(df_chn['ooh_S'], 0.4)
dfcorr['print_SDecay0.4'] = adstockGeometric(df_chn['print_S'], 0.4)
dfcorr['search_SDecay0.3'] = adstockGeometric(df_chn['search_S'], 0.3)
dfcorr['facebook_SDecay0.3'] = adstockGeometric(df_chn['facebook_S'], 0.3)


dfmmm_final=pd.concat([df['revenue'],dfcorr],axis=1)
NPcorr = dfmmm_final.corr()['revenue'][:]
NPcorr = pd.DataFrame(NPcorr).reset_index()
NPcorr.columns=['var','corr']

NPcorr = NPcorr.sort_values(by=['corr'], ascending = False)

x_tf=NPcorr['var'].tolist()
x_tf.remove('revenue')

dftest = pd.concat([df['DATE'],dfcorr,df['competitor_sales_B']],axis=1)
dftest = pd.DataFrame(dftest).set_index('DATE')
df_temp = pd.DataFrame(df).set_index('DATE')

#['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S']
paper_cofe = np.array([[0.418,0.31,0.334,0.642,0.689,6.999]])
xgb_cofe = np.array([[0.3276,0.120162,0.218303,0,0.333935,0]])
coef_ = paper_cofe*xgb_cofe+paper_cofe

x_tf.append('competitor_sales_B')
train_x = dftest[x_tf]['2015-11-23':'2019-09-09']
train_y = df_temp[['revenue']]['2015-11-23':'2019-09-09']

test_x = dftest[x_tf]['2019-10-14':'2019-11-11']
test_y = df_temp[['revenue']]['2019-10-14':'2019-11-11']


Lambdas = np.logspace(-5,2,200)
rigdeCV = RidgeCV(alphas=Lambdas,scoring='neg_mean_squared_error')
model = rigdeCV.fit(train_x,train_y)
model_pred = rigdeCV.predict(test_x)
model_r2=r2_score(test_y,model_pred)
model_list = np.append(model.coef_,model.intercept_)
model_list = np.append(model_list,model_r2)
model_list = np.append(model_list,round(abs(test_y.values.sum() - model_pred.sum())/test_y.values.sum()*100,2))

model_name=X.copy()
model_name.extend(['competitor_sales_B','intercept_','R2','mape'])
model_data=pd.DataFrame(model_list,index=model_name)
print(model_data)
model_data.columns=['coef']
model_data1=model_data.sort_values('coef',ascending=False)
model_data1.to_csv('rigde_coef&other.csv')

paper_cofe = np.array([[0.418,0.31,0.934,0.642,0.689,6.999]])
xgb_cofe = np.array([[0.3276,0.120162,0.218303,0,0.333935,0]])
# paper_cofe = np.array([[0.253064,0.301505,0.315302,0.440859,0.668056,0.145234]])
# xgb_cofe = np.array([[0.3276,0.120162,0.218303,0,0.333935,0]])
model.coef_ = paper_cofe*xgb_cofe+paper_cofe

y_hat = np.sum(train_x*coef_,axis=1).values
model.intercept_ = np.median(train_y.values.reshape(-1)-y_hat)
# plt.plot(train_y.values.reshape(-1),'r',label='real')
# plt.plot(model.intercept_+y_hat,'g',label='simulate')
# plt.plot(y_hat,'b',label='only_channel')
# plt.ylabel("revenue")
# plt.xlabel("时间")
# plt.title("训练集上的拟合情况")
# plt.legend()
# plt.show()

y_hat = np.sum(dftest*coef_,axis=1).values
model_pred = model.intercept_+y_hat
y = df_temp[['revenue']]
model_r2=r2_score(y,model_pred)
model_list = np.append(model.coef_,model.intercept_)
model_list = np.append(model_list,model_r2)
model_list = np.append(model_list,round(abs(y.values.sum() - model_pred.sum())/y.values.sum()*100,2))
model_name=X.copy()
model_name.extend(['competitor_sales_B','intercept_','R2','mape'])
model_data=pd.DataFrame(model_list,index=model_name)
print(model_data)
model_data.columns=['coef']
model_data1=model_data.sort_values('coef',ascending=False)
model_data1.to_csv('rigde_coef&other_finall.csv')

plt.subplot(2,1,1)
plt.plot(y.values,'r-',label='real')
plt.plot(model_pred,'g--',label='RidgeMMM_simulate')
plt.ylabel("revenue")
plt.xlabel("时间")
plt.title("全部时间上的拟合情况")
plt.legend()


model_pred = model_pred[-5:]
model_r2=r2_score(test_y,model_pred)
model_list = np.append(model.coef_,model.intercept_)
model_list = np.append(model_list,model_r2)
model_list = np.append(model_list,round(abs(test_y.values.sum() - model_pred.sum())/test_y.values.sum()*100,2))

model_name=X.copy()
model_name.extend(['competitor_sales_B','intercept_','R2','mape'])
model_data=pd.DataFrame(model_list,index=model_name)
print(model_data)
model_data.columns=['coef']
model_data1=model_data.sort_values('coef',ascending=False)
model_data1.to_csv('rigde_coef&other_finpred.csv')
plt.subplot(2,1,2)
plt.plot(test_y,'r-',label='real')
plt.plot(model_pred,'g--',label='RidgeMMM_simulate')
plt.ylabel("revenue")
plt.xlabel("时间")
plt.title("'2019-10-14'至'2019-11-11'的预测情况")
plt.legend()


# model_pred = rigdeCV.predict(dftest)
# test_y = df_temp[['revenue']]
# model_r2=r2_score(test_y,model_pred)
# model_list = np.append(model.coef_,model.intercept_)
# model_list = np.append(model_list,model_r2)
# model_list = np.append(model_list,round(abs(test_y.values.sum() - model_pred.sum())/test_y.values.sum()*100,2))
# model_name=X.copy()
# model_name.extend(['intercept_','R2','mape'])
# model_data=pd.DataFrame(model_list,index=model_name)
# print(model_data)
# model_data.columns=['coef']
# model_data1=model_data.sort_values('coef',ascending=False)
# model_data1.to_csv('rigde_coef&other_finall.csv')
#
# plt.subplot(2,1,1)
# plt.plot(test_y,'r-',label='real')
# plt.plot(y_hat,'g--',label='RidgeMMM_simulate')
# plt.ylabel("revenue")
# plt.xlabel("时间")
# plt.title("全部时间上的拟合情况")
# plt.legend()

plt.tight_layout()
plt.show()