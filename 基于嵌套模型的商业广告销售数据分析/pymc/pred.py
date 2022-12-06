import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3 import *
import arviz as az

df_in = pd.read_csv("de_simulated_data.csv")
df_temp = pd.DataFrame(df_in).set_index('DATE')
df_in = df_temp['2015-11-23':'2019-09-09']
df_test = df_temp['2019-10-14':'2019-11-11']
delay_channels = ['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S']
non_lin_channels = ['competitor_sales_B']



def logistic_function(x_t, mu=0.1):
    return (1 - np.exp(-mu.values * x_t)) / (1 + np.exp(-mu.values * x_t))


import theano.tensor as tt


def geometric_adstock_tt(x_t, alpha=0, L=2):
    # w = tt.as_tensor_variable([tt.power(alpha, i) for i in range(L)])
    # xx = tt.stack([tt.concatenate([tt.zeros(i), x_t[:x_t.shape[0] - i]]) for i in range(L)])
    w = [alpha**i for i in range(L)]
    xx = np.stack([np.concatenate([np.zeros(i), x_t[:x_t.shape[0] - i]])for i in range(L)])
    y = np.sum(w // np.sum(w)*xx,axis=0)
    # y = tt.dot(w // tt.sum(w), xx)
    return y


az_summary = pd.read_excel('azsummary.xlsx')
df_pred = az_summary[['chn','mean']]

pred_mean = []
for channel_name in delay_channels:
    xx = df_test[channel_name].values
    alpha = df_pred[df_pred['chn'].str.contains('alpha_' + channel_name)]['mean']
    channel_mu = df_pred[df_pred['chn'].str.contains('mu_' + channel_name)]['mean']
    channel_b = df_pred[df_pred['chn'].str.contains('beta_' + channel_name)]['mean']
    # pred_mean.append(logistic_function(geometric_adstock_tt(xx, alpha), channel_mu) * channel_b.values)
    pred_mean.append(xx * channel_b.values)

for channel_name in non_lin_channels:
    xx = df_test[channel_name].values
    channel_mu = df_pred[df_pred['chn'].str.contains('mu_' + channel_name)]['mean']
    channel_b = df_pred[df_pred['chn'].str.contains('beta_' + channel_name)]['mean']
    pred_mean.append(logistic_function(xx, channel_mu) * channel_b.values)
    # pred_mean.append(xx * channel_b.values)

sigma = df_pred[df_pred['chn'].str.contains('sigma')]['mean'].values
pred_mean.append(sigma)
pred = sum(pred_mean)
mape = np.abs(sum(pred)-sum(df_test[['revenue']].values))/sum(df_test[['revenue']].values)
print()