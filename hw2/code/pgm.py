import pandas as pd
import numpy as np
from random import shuffle
from numpy.linalg import inv
from math import floor, log
import os
import argparse

output_dir = "../output"


# 处理原始数据
def dataProcess_X(rawData):

    # 筛选DataFrame的横向索引
    if "income" in rawData.columns:     # 训练集
        Data = rawData.drop(["sex", 'income'], axis = 1)
    else:                               # 测试集
        Data = rawData.drop(["sex"], axis = 1)

    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"]    # 读取非数字的column
    # print(listObjectColumn)
    listNonObjedtColumn = [x for x in list(Data) if x not in listObjectColumn]          # 数字的column
    # print(listNonObjedtColumn)

    ObjectData = Data[listObjectColumn] # 非数字的数据
    NonObjectData = Data[listNonObjedtColumn] # 数字的数据

    # 添加性别，female = 1, male = 0
    NonObjectData.insert(0, "sex", (rawData["sex"] == " Female").astype(np.int))

    # 非数字的属性变为独热编码
    ObjectData = pd.get_dummies(ObjectData)

    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data_x = Data.astype("int64")
 
    # normalize
    Data_x = (Data_x - Data_x.mean()) / Data_x.std()

    print(Data_x)
    return Data_x


def dataProcess_Y(rawData):
    df_y = rawData['income']
    Data_y = pd.DataFrame((df_y == ' >50K').astype("int64"), columns=['income'])

    # print(Data_y)
    return Data_y

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))

    return np.clip(res, 1e-8, (1-(1e-8)))


if __name__ == "__main__":
    trainData = pd.read_csv("../data/train.csv")
    testData = pd.read_csv("../data/test.csv")
    ans = pd.read_csv("../data/sample_submission.csv")

    # dataProcess_X(trainData)
    dataProcess_Y(trainData)