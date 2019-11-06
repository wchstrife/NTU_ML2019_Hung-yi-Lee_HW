# -*- coding:UTF-8 -*-
import numpy as np
import pandas as pd

# 指定路径
path = "./data/"

# 加载数据
train = pd.read_csv(path + 'train.csv', encoding='utf-8')
test = pd.read_csv(path + 'test.csv', encoding='gbk')
print(train)
train = train[train['observation'] == 'PM2.5']
print(train)

