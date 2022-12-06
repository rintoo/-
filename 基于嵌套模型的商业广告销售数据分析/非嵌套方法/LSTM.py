import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Dropout
import matplotlib.pyplot as plt
import tensorflow as tf

# 在0-999生成1000个值
x = np.linspace(0,999,1000)
# 主要生成y为周期曲线用于后续预测
y = np.sin(x*2*3.1415926/70)
# 可视化展示构建曲线
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title("sin")
plt.plot(y, color='#800080')
plt.show()

# 将数据分成两部分，训练集由前930个值构成，测试集由最后70个值构成
train_y = y[:-70]
test_y = y[-70:]

# 创建滑窗数据集
def create_data_seq(seq,time_window):
    x = []
    y = []
    l = len(seq)
    for i in range(l-time_window):
        x_tw = seq[i:i+time_window]
        y_tw = seq[i+time_window:i+time_window+1]
        x.append(x_tw)
        y.append(y_tw)
    return np.expand_dims(np.array(x),axis=2), np.array(y)
# 设置滑窗个数
time_window = 30
# 训练集
train_X, train_Y = create_data_seq(train_y,time_window)
# 测试集
test_X, test_Y = create_data_seq(test_y,time_window)

# 设置gpu内存自增长
gpus = tf.config.experimental.list_physical_devices('GPU')

model = Sequential()
model.add(LSTM(30, input_shape=(train_X.shape[1], train_X.shape[2]))) #lstm层，设置30个神经元
model.add(Dropout(0.5)) #随机丢弃50%
model.add(Dense(1,activation='tanh')) #激活函数为tanh，线性层
model.compile(loss='mae', optimizer='adam') #损失函数为mae,优化函数为adam

model.fit(train_X, train_Y, epochs=10, batch_size=8, verbose=2)

# 训练集预测，可视化查看拟合效果
plt.plot(model.predict(test_X).reshape(-1),label='pre')
plt.plot(test_Y, color='#800080',label='true')
plt.legend()
plt.show()
