import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load 'train.csv'
# train.csv 的資料為 12 個月中，每個月取 20 天，每天 24 小時的資料(每小時資料有 18 個 features)。
data = pd.read_csv('./train.csv', encoding='big5')

# Preprocessing
# 取需要的數值部分，將 'RAINFALL' 欄位全部補 0
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

# Extract Features (1)

# 將原始 4320 * 18 的資料依照每個月分重組成 12 個 18 (features) * 480 (hours) 的資料。
month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24: (day + 1) * 24] = raw_data[18 * (20 * month + day): 18 * (20 * month + day + 1), :]
    month_data[month] = sample

# 为什么会有471的data？？

# Extract Features (2)
# 每個月會有 480hrs，每 9 小時形成一個 data，每個月會有 471 個 data，故總資料數為 471 * 12 筆，而每筆 data 有 9 * 18 的 features (
# 一小時 18 個 features * 9 小時)。 對應的 target 則有 471 * 12 個(第 10 個小時的 PM2.5) 训练数据的输入x和答案y
x = np.empty([12 * 471, 18 * 9], dtype=float)
y = np.empty([12 * 471, 1], dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1,
                                                                                                                     -1)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9]  # value

# Normalize (1)

mean_x = np.mean(x, axis=0)  # 18 * 9
std_x = np.std(x, axis=0)  # 18 * 9
for i in range(len(x)):  # 12 * 471
    for j in range(len(x[0])):  # 18 * 9
        if std_x[j] != 0:
            x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]

# Split Training Data Into "train_set" and "validation_set"
# 分出自测数据
import math

# 避免overfitting,一般遇到的过拟合应该是0.8(训练集损失)与2.0(验证集损失)这种不在一个量级的损失比
# https://juejin.im/post/5be5b0d7e51d4543b365da51
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8):, :]
y_validation = y[math.floor(len(y) * 0.8):, :]

# Training

# 训练模型的次数
train_num = 4
iteration_num = 1000
loss_array = np.empty([train_num, iteration_num], dtype=float)


def train(x, y, learning_rate, model_id):
    print('--------' + str(model_id) + "--------")
    dim = 18 * 9 + 1
    w = np.zeros([dim, 1])
    x = np.concatenate((np.ones([int(12 * 471 * 0.8), 1]), x), axis=1).astype(float)
    # learning_rate = 100
    iter_time = iteration_num
    adagrad = np.zeros([dim, 1])
    eps = 0.0000000001
    for t in range(iter_time):
        loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2)) / 471 / 12)  # rmse
        loss_array[model_id - 1, t] = loss
        if t % 100 == 0:
            print(str(t) + ":" + str(loss))
        gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y)  # dim*1
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    np.save('weight_md' + str(model_id) + '.npy', w)


# Testing
# 載入 test data，並且以相似於訓練資料預先處理和特徵萃取的方式處理，使 test data 形成 240 個維度為 18 * 9 + 1 的資料。
testdata = pd.read_csv('./test.csv', header=None, encoding='big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18 * 9], dtype=float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_x[j] != 0:
            test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

# 训练模型
# 模型增加时，记得增加color的颜色值
train(x_train_set, y_train_set, 100, 1)
train(x_train_set, y_train_set, 0.37, 2)
train(x_train_set, y_train_set, 0.2, 3)
train(x_train_set, y_train_set, 0.05, 4)

# 画图
x_axis = np.linspace(0, iteration_num, iteration_num)
plt.figure()
color = ('red', 'orange', 'blue', 'green')
for i in range(train_num):
    plt.plot(x_axis, loss_array[i, 0:iteration_num], color=color[i])
plt.show()

# 计算预测结果
ans_y_array = np.empty([train_num, int(len(x) * 0.2 + 1)], dtype=float)
for i in range(train_num):
    w = np.load('weight_md' + str(i + 1) + '.npy')
    ans_y = np.dot(test_x, w)
    for j in range(240):
        ans_y_array[i][j] = ans_y[j][0]

# Save Prediction to CSV File
# 生成提交文件

import csv

# 提取第几个模型的数据，默认第一个
num = 0
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y_array[num][i]]
        csv_writer.writerow(row)
        print(row)
