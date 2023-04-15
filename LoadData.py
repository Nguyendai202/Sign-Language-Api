import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from random import shuffle 
# đưa véc tơ label về ma trận onehotcoding 


def load_data ():
    train_data = pd.read_csv('data/sign_mnist_train.csv')
    test_data = pd.read_csv('data/sign_mnist_test.csv')
    X_train = train_data.iloc[:,1:].values # chọn tất cả hàng và cột bắt đầu từ cột thứ 2 và chuyển sang numpy arr = values
    Y_train = train_data['label'].values
    X_test = test_data.iloc[:,1:].values
    Y_test = test_data['label'].values
    X_train = np.array(X_train).reshape((-1,1,28,28)).astype(np.uint8)/255.0
    X_test = np.array(X_test).reshape((-1,1,28,28)).astype(np.uint8)/255.0
    Y_train = to_categorical(Y_train,25).astype(np.uint8)
    # gán từng ma trận data kèm theo nhãn của nó 
    training_data =[]
    for i ,data in enumerate(X_train): # i là số thứ tự tương ứng 
        train_label = Y_train[i]
        training_data.append([np.array(data),np.array(train_label)])
    shuffle(training_data)# xáo trộn 
    testing_data = []
    for i, data in enumerate(X_test):
        testing_data.append([np.array(data), i+1])
    return training_data,testing_data,Y_train,Y_test

trainning_data , testing_data ,Y_train,Y_test = load_data ()











