from pymc3 import *
import pymc3 as pm
import numpy as np
import theano.tensor as tt
import pandas as pd


# def adstockGeometric(x, theta):
#     x_decayed = [0 for i in range(len(x))]
#     x_decayed[0] = x[0]
#     for i in range(len(x)-1):
#         x_decayed[i+1] = x[i+1] + theta * x_decayed[i]
#     return x_decayed
#
# def hill(x_normalized, alpha, gamma):
#     #dt = pd.Series(np.array([6, 47, 49, 15, 42, 41, 7, 39, 43, 40, 36])
#     # 直接进行分数线划分
#     gammaTrans = gamma*(max(x_normalized)-min(x_normalized))+min(x_normalized)
#     x_ref = x_normalized**alpha/(x_normalized**alpha+gammaTrans**alpha)
#     return x_ref
def logistic_function(x_t, mu=0.1):
    return (1 - np.exp(-mu * x_t)) / (1 + np.exp(-mu * x_t))


import theano.tensor as tt


def geometric_adstock_tt(x_t, alpha=0, L=12, normalize=True):
    w = tt.as_tensor_variable([tt.power(alpha, i) for i in range(L)])
    xx = tt.stack([tt.concatenate([tt.zeros(i), x_t[:x_t.shape[0] - i]]) for i in range(L)])

    if not normalize:
        y = tt.dot(w, xx)
    else:
        y = tt.dot(w // tt.sum(w), xx)
    return y


df_fb = pd.read_csv('de_simulated_data.csv')
df_fb = df_fb.drop(labels=['facebook_I','search_clicks_P'],axis=1)
chn = df_fb.drop(labels=['competitor_sales_B','revenue'],axis=1)
control_Z = df_fb['competitor_sales_B']
sale = df_fb['revenue']

with pm.Model() as model_test:
    # intercept = pm.Normal('intercept',0,sigma=1)
    # beta = pm.HalfNormal('beta',sd=1,shape=5)
    # theta = pm.Beta('theta',alpha=1,beta=3,shape=5)
    # c_beta = pm.Normal('control_beta',sd=1)
    # sigma = pm.Exponential('sigma',10)
    # alpha = pm.Gamma('alpha',alpha=3,beta=1,shape=5)
    # gamma = pm.Gamma('gamma',alpha=3,beta=1,shape=5)
    intercept = pm.Normal('intercept', 0, sigma=1)
    beta = pm.HalfNormal('beta', sd=1, shape=5)
    half_sat = pm.Gamma('half_sat', alpha=3, beta=1, shape=5)
    alpha = pm.Beta('alpha', alpha=1, beta=3, shape=5)
    c_beta = pm.Normal('control_beta', sd=1)
    sigma = pm.Exponential('sigma', 10)

    coef_x = []
    for i in range(5):
        # coef_x.append(beta[i] * geometric_adstock_tt(logistic_function(chn.iloc[:,i].values,half_sat),alpha[i]))
        coef_x.append(beta[i] * logistic_function(geometric_adstock_tt(chn.iloc[:,i].values, alpha), half_sat[i]))
    mu = intercept + coef_x + c_beta*control_Z
    likelihood = pm.Normal('y_obs',mu=mu,sigma=sigma,observed=sale)
    trace = pm.sample()
    az.summary(trace)