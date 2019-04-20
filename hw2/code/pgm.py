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

    # sex 只有两个属性，先drop
    if "income" in rawData.columns:
        Data = rawData.drop(["sex", 'income'], axis = 1)
    else:
        Data = rawData.drop(["sex"], axis = 1)

    listObjectColumn = [col for col in Data.columns if Data[col].dtypes == "object"] #读取非数字的column
    # print(listObjectColumn)
    listNonObjedtColumn = [x for x in list(Data) if x not in listObjectColumn] #数字的column
    # print(listNonObjedtColumn)

    ObjectData = Data[listObjectColumn] # 非数字的数据
    NonObjectData = Data[listNonObjedtColumn] # 数字的数据

    # 添加性别，female = 0, male = 1
    NonObjectData.insert(0, "sex", (rawData["sex"] == " Female").astype(np.int))

    # 非数字的属性变为独热编码
    ObjectData = pd.get_dummies(ObjectData)

    Data = pd.concat([NonObjectData, ObjectData], axis=1)
    Data_x = Data.astype("int64")

    #normalize
    Data_x = (Data_x - Data_x.mean()) / Data_x.std()

    return Data_x
    
    


if __name__ == "__main__":
    trainData = pd.read_csv("../data/train.csv")
    print(trainData)
    dataProcess_X(trainData)