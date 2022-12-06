from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_excel('./spss多层感知机/仅渠道数据的多层感知机预测.xlsx')

y=df['revenue']
y_hat = df['PredictedValue']
print('all:',r2_score(y,y_hat))
df = df.iloc[199:,:]
y=df['revenue']
y_hat = df['PredictedValue']
print('pred:',r2_score(y,y_hat))