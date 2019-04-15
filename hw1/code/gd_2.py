# -*- coding: utf-8 -*-
## Kaggle soce ada + DG with ans_2: 5.79589
## Kaggle soce SDG with ans_3: 13.31860

import csv, os
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import random
import math
import sys


def ada(X, Y, w, eta, iteration, lambdaL2):
    s_grad = np.zeros(len(X[0]))
    list_cost = []
    for i in range(iteration):
        hypo = np.dot(X, w)
        loss = hypo - Y
        cost = np.sum(loss**2) / len(X)
        list_cost.append(cost)

        grad = np.dot(X.T, loss) / len(X) + lambdaL2 * w
        s_grad += grad**2
        ada = np.sqrt(s_grad)
        w = w - eta * grad / ada
    return w, list_cost



def SGD(X, Y, w, eta, iteration, lambdaL2):
    list_cost = []
    for i in range(iteration):
        hypo = np.dot(X, w)
        loss = hypo - Y
        cost = np.sum(loss**2)/len(X)
        list_cost.append(cost)

        rand = np.random.randint(0, len(X))     # SGD 和 GD 区别在于SGD随机一个数据做求导
        grad = X[rand] * loss[rand] / len(X) + lambdaL2 * w     # 不需要用矩阵计算全部的数据
        w = w - eta * grad
    return w, list_cost

def GD(X, Y, w, eta, iteration, lambdaL2):
    list_cost = []
    for i in range(iteration):
        hypo = np.dot(X, w)
        loss = hypo - Y
        cost = np.sum(loss**2)/len(X)
        list_cost.append(cost)

        grad = np.dot(X.T, loss)/len(X) + lambdaL2 * w
        w = w - eta*grad
    return w, list_cost

# 每一个维度存一种污染物的数据
data = []
for i in range(18):
    data.append([])


# read data
n_row = 0
text = open('data/train.csv', 'r', encoding = 'big5')
rows = csv.reader(text, delimiter = ",")
for r in rows:
    if n_row != 0:
        for i in range(3, 27):
            if r[i] != "NR":
                data[(n_row - 1) % 18].append(float(r[i]))
            else:
                data[(n_row - 1) % 18].append(float(0))
    n_row = n_row + 1
text.close

print(len(data))
print(len(data[0]))

# parse data to trainX and trainY
x = []
y = []
for i in range(12):
    for j in range(471):
        x.append([])
        for t in range(18):
            for s in range(9):
                x[471*i + j].append(data[t][480*i+j+s])
        y.append(data[9][480*i+j+9])
trainX = np.array(x)
trainY = np.array(y)
print("train_x:", trainX.shape)
print("train_y:", trainY.shape)


# parse test data 
test_x = []
n_row = 0
text = open('data/test.csv', 'r')
rows = csv.reader(text, delimiter = ",")

for r in rows:
    if n_row % 18 == 0:
        test_x.append([])
        for i in range(2, 11):
            test_x[n_row//18].append(float(r[i]))
    else:
        for i in range(2, 11):
            if r[i] != "NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row + 1
text.close
test_x = np.array(test_x)
print("test_x:", test_x.shape)


# parse anser
'''
ans_y = []
n_row = 0
text = open('data/ans.csv', "r")
row = csv.reader(text, delimiter=",")

for r in row:
    ans_y.append(r[1])

ans_y = ans_y[1:]
ans_y = np.array(list(map(int, ans_y)))
'''

# add bias
trainX = np.concatenate( ( np.ones((trainX.shape[0], 1)), trainX ), axis = 1 )
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)
print("train_x:", trainX.shape)
print("test_x:", test_x.shape)


# train data
w = np.zeros(len(trainX[0]))
w_sgd, cost_list_sgd = SGD(trainX, trainY, w, eta=0.0001, iteration=20000, lambdaL2 = 0)
w_ada, cost_list_ada = ada(trainX, trainY, w, eta=1, iteration=20000, lambdaL2 = 0)


#output testdata
y_ada = np.dot(test_x, w_ada)
y_sgd = np.dot(test_x, w_sgd)


#csv format
ans = []
for i in range(len(test_x)):
    ans.append(["id_" + str(i)])
    a = np.dot(w_sgd, test_x[i])
    ans[i].append(a)

filename = "result/ans_3.csv"
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()

