# -*- coding:UTF-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import csv

# 数据路径
path = "../data/"

def data_process():

    # 加载数据
    train = pd.read_csv(path + "train.csv", encoding='big5')
    test = pd.read_csv(path + "test.csv", encoding='big5')

    train = train[train['測項'] == 'PM2.5']
    test = test[test['AMB_TEMP'] == 'PM2.5']

    # 去掉多余的字段
    train = train.drop(['日期', '測站', '測項'], axis=1)
    test_x = test.iloc[:, 2:]


    train_x = []
    train_y = []

    for i in range(15):
        # 总共9列作为输入
        x = train.iloc[:, i:i + 9]    
        x.columns = np.array(range(9))
        # 第10列作为输出
        y = train.iloc[:, i + 9]
        y.columns = np.array(range(1))
        train_x.append(x)
        train_y.append(y)

    train_x = pd.concat(train_x)    # (3600, 9)
    train_y = pd.concat(train_y)    # (3600, 1)

    train_y = np.array(train_y, float)
    test_x = np.array(test_x, float)

    # 归一化
    ss = StandardScaler()
    ss.fit(train_x)
    train_x = ss.transform(train_x)
    ss.fit(test_x)
    test_x = ss.transform(test_x)

    return train_x, train_y, test_x

# 损失函数
def loss_funtion(weight_W, data_X, y):
    return np.sum((y - data_X.dot(weight_W)) ** 2) / len(y)

# 损失函数的导数
def der_loss_function(weight_W, data_X, y):
    return data_X.T.dot(data_X.dot(weight_W) - y) * 2 / len(y)


# 更新参数，训练模型
def train_data(train_x, train_y, epoch):

    learning_rate = 0.01       # 学习率

    data_X = np.hstack([np.ones((len(train_x), 1)), train_x])
    init_w = np.zeros(data_X.shape[1])

    for i in range(epoch):
        gradient = der_loss_function(init_w, data_X, train_y)
        init_w = init_w - learning_rate * gradient

    return init_w

# 预测test data的值
def predict(predict_x, w):
    test_x = np.hstack([np.ones((len(predict_x), 1)), predict_x])
    return test_x.dot(w)

# 评价预测值和真实值得差距
def cal_score(y_true, y_predict):
    MSE = np.sum((y_true - y_predict) ** 2) / len(y_true)
    return 1 - MSE / np.var(y_true)


if __name__ == "__main__":
    train_x, train_y, test_x = data_process()
    weight = train_data(train_x, train_y, 10000)

    # 计算训练集的得分
    train_x_pre = predict(train_x, weight)
    score = cal_score(train_y, train_x_pre)
    print(score)

    test_y = predict(test_x, weight)

    # 结果保存
    sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv', engine='python', encoding='gbk')
    sampleSubmission['value'] = test_y
    sampleSubmission.to_csv("../result/ans.csv", index=False)
    
    
