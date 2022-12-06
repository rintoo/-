import numpy as np
import pandas as pd
import pymc3 as pm
from pymc3 import *
import arviz as az

# d = {}
# for i in range(1, 13, 1):
#     d[f"channel_{i}"] = np.random.uniform(0, 1, 12)
# df_in = pd.DataFrame(d)
# df_in["y"] = np.random.uniform(0, 1, 12)
# delay_channels = ["channel_1", "channel_2", "channel_3", "channel_10"] # channels that can have both decay and saturation effects
# non_lin_channels = ["channel_4", "channel_5", "channel_6", "channel_7",
#                     "channel_12", "channel_11", "channel_9", "channel_8"]
df_in = pd.read_csv("de_simulated_data.csv")
delay_channels = ['tv_S', 'ooh_S', 'print_S', 'search_S', 'facebook_S']
non_lin_channels = ['competitor_sales_B']

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


with Model() as model:
    response_mean = []
    # channels that can have decay and saturation effects.
    for channel_name in delay_channels:
        xx = df_in[channel_name].values
        print(f"Adding delayed channel: {channel_name}")
        channel_b = HalfNormal(f"beta_{channel_name}", sd=5)
        alpha = Beta(f"alpha_{channel_name}", alpha=3, beta=3)
        channel_mu = Gamma(f"mu_{channel_name}", alpha=3, beta=1)
        # we transform the marketing spend with the adstock and then that transformed version is fed to the logistic
        # reach function that models our saturation.
        response_mean.append(logistic_function(geometric_adstock_tt(xx, alpha), channel_mu) * channel_b)

    # channels that can have decay and saturation effects
    for channel_name in non_lin_channels:
        xx = df_in[channel_name].values

        print(f"Adding non linear logistic channel: {channel_name}")
        channel_b = HalfNormal(f"beta_{channel_name}", sd=5)

        # logistic reach curve
        channel_mu = Gamma(f"mu_{channel_name}", alpha=3, beta=1)
        response_mean.append(logistic_function(xx, channel_mu) * channel_b)

    # Continuous control variables
    #     if control_vars:
    #         for channel_name in control_vars:
    #             x = df_in[channel_name]
    #             print(f"adding control: {channel_name}")
    #             control_beta = Normal(f"beta_{channel_name}", s=0.25)
    #             channel_contrib = control_beta * x
    #             response_mean.append(channel_contrib)

    #     # Categorical control variables.
    #     if index_vars:
    #         for var_name in index_vars:
    #             shape = len(df_in[var_name].unique())
    #             x = df_in[var_name].values

    #             print(f"adding index variable : {var_name}")
    #             ind_beta = Normal(f"beta_{var_name}", sd=5, shape=shape)
    #             channel_contrib = ind_beta[x]
    #             response_mean.append(channel_contrib)

    sigma = Exponential("sigma", 10)
    likelihood = Normal("revenue", mu=sum(response_mean), sd=sigma, observed=df_in["revenue"].values)
def main():
    with model:
        start = pm.find_MAP()
        step = pm.NUTS(scaling=start)
        trace = pm.sample(5000, step, start=start,chains=2, progressbar=True)
        # trace = pm.sample(10000, tune=1000, chains=2)
    az.summary(trace)

if __name__=='__main__': #不加这句就会报错
    main()
