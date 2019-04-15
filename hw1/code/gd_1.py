# -*- coding: utf-8 -*-
## Kaggle soce: 8.07501


import sys
import numpy as np
import pandas as pd
import csv
from io import BytesIO
import math

def load_train_data(train_name):
    data = []
    for i in range(18):
        data.append([])

    # print(data)

    n_row = 0
    text = open(train_name, 'r', encoding='big5')
    reader = csv.reader(text, delimiter=",")

    for row in reader:
        if n_row != 0:
            for i in range(3, 27):
                if(row[i] != 'NR'):
                    data[(n_row - 1) % 18].append(float(row[i]))
                else:
                    data[(n_row - 1) % 18].append(float(0))
        n_row = n_row + 1

    text.close

    return data

def extract_features(data):
    data = np.array(data)
    x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float)
    y = np.empty(shape = (12 * 471 , 1),dtype = float)

    for month in range(12):
        for day in range(20):
            for hour in range(24):
                if day == 19 and hour > 14:
                    continue
                x[month * 471 + day * 24 + hour, :] = data[:, month * 12 + day * 24 + hour : month * 12 + day * 24 + hour + 9].reshape(1, -1)
                y[month * 471 + day * 24 + hour, 0] = data[9, month * 12 + day * 24 + hour + 9]

    x = np.array(x)
    y = np.array(y)

    print(x.shape[0])
    print(x.shape[1])
    print(y.shape[1])

    # normalization
    mean = np.mean(x, axis = 0) # 对各列求平均值
    std = np.std(x, axis=0)     # 对各列求标准差
    for i in range(x.shape[0]): # 12 * 471
        for j in range(x.shape[1]): # 18 * 9
            if not std[j] == 0 :
                x[i][j] = (x[i][j]- mean[j]) / std[j]

    return x, y

def training(x ,y):
    dim = x.shape[1] + 1 # 18 * 9 + 1
    w = np.zeros(shape = (dim, 1))  # [18*9+1 , 1]
    x = np.concatenate( (np.ones((x.shape[0], 1)),x), axis = 1 ).astype(float)
    print("x shape:", x.shape)
    learning_rate = np.array([[200]] * dim)     # [163, 1] data = 200
    print("learning_rate shape:", learning_rate.shape)
    adagrad_sum = np.zeros(shape = (dim, 1 ))   # [163, 1] data = 0

    for T in range(10000):
        
        if(T % 100 == 0):
            print("T=",T)
            print("Loss:",np.power(np.sum(np.power(x.dot(w) - y, 2 ))/ x.shape[0],0.5))

        gradient = (-2) * np.transpose(x).dot(y-x.dot(w))
        adagrad_sum += gradient ** 2
        w = w - learning_rate * gradient / (np.sqrt(adagrad_sum) + 0.0005)

    np.save('weight.npy', w)     ## save weight

def predict():
    w = np.load('weight.npy')
    test_raw_data = np.genfromtxt("test.csv", delimiter=',')
    test_data = test_raw_data[:, 2: ]
    where_are_NaNs = np.isnan(test_data)
    test_data[where_are_NaNs] = 0 
    print("read test.csv success!")
    
    test_x = np.empty(shape = (240, 18 * 9),dtype = float)

    for i in range(240):
        test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1) 

     ## Normalization
    mean = np.mean(test_x, axis = 0) # 对各列求平均值
    std = np.std(test_x, axis=0)     # 对各列求标准差
    for i in range(test_x.shape[0]):       
        for j in range(test_x.shape[1]):
            if not std[j] == 0 :
                test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]

    test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
    answer = test_x.dot(w)

    f = open("result.csv","w+")
    w = csv.writer(f,delimiter=',',lineterminator='\n')
    title = ['id','value']
    w.writerow(title) 
    for i in range(240):
        content = ['id_'+str(i),answer[i][0]]
        w.writerow(content) 
    f.close


def main():
    # data = load_train_data("train.csv")
    # x, y = extract_features(data)
    # training(x, y)
    # print("finish!")
    predict()

                    




    

if __name__ == '__main__':
    main()
    







