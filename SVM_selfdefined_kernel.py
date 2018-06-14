# 个人定义与实现的SVM核
# 课程：Machine Learning CS420 
# By: 万俊成

import pandas as pd
from pandas import Series,DataFrame
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from libsvm.python.svm import *
from libsvm.python.svmutil import *
import numpy as np
from PIL import Image
import math
n_train = 60000 #载入训练集大小
n_test = 10000  #载入测试集大小
fig_w = 45      #图片长宽大小

train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
train_label = np.fromfile("mnist_train_label",dtype=np.uint8)

test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
test_label = np.fromfile("mnist_test_label",dtype=np.uint8)

#数据预处理，改变一下形状
train_data1 = list(train_data.reshape(n_train,fig_w*fig_w))
train_label1 = list(train_label)

test_data1 = list(test_data.reshape(n_test,fig_w*fig_w))
test_label1 = list(test_label)

size_train=6000 #实际训练数据集的大小
size_test=10000 #实际测试数据集的大小

#Rational Quadratic Kernel
const=10  #const 是参数之一
n_dim=2025

my_SVM=SVC(kernel='precomputed')

tmp=np.empty((size_train,size_train),np.int)
for i in range(0,size_train):
    for j in range(0,size_train):
        sum=0
        for k in range(0,n_dim):
            sum=sum+(train_data1[i][k]-train_data1[j][k])**2
        tmp[i][j]=const/(sum+const)
k_train_data=tmp

my_SVM.fit(k_train_data2[0:size_train], train_label1[0:size_train])

for i in range(0,size_test):
    for j in range(0,size_train):
        sum1=0
        for k in range(0,n_dim):
            sum1=sum1+(test_data1[i][k]-train_data1[j][k])**2
        tmp[i][j]=const/(sum1+const)
k_test_date2=tmp

y_pred=my_SVM.predict(k_test_date2[0:size_test])
score_SVM=accuracy_score(test_label1[0:size_test], y_pred)
print("The accuracy score of my kernel1-Rational Quadratic Kernel:", score_SVM)

#Spherical Kernel
const=10
n_dim=2025
delta=100 #delta不能太小，太小会数值过大

my_SVM=SVC(kernel='precomputed')

tmp=np.empty((size_train,size_train),np.int)
for i in range(0,size_train):
    for j in range(0,size_train):
        sum1=0
        sum2=0
        for k in range(0,n_dim):
            ssum1=sum1+(train_data1[i][k]-train_data1[j][k])**2
            sum2=sum2+abs(train_data1[i][k]-train_data1[j][k])
        tmp[i][j]=1-1.5*sum2/delta+0.5*(sum1/delta)**3
k_train_data3=tmp

my_SVM.fit(k_train_data3[0:size_train], train_label1[0:size_train])

for i in range(0,size_test):
    for j in range(0,size_train):
        sum1=0
        sum2=0
        for k in range(0,n_dim):
            sum1=sum1+(test_data1[i][k]-train_data1[j][k])**2
            sum2=sum2+abs(test_data1[i][k]-train_data1[j][k])
        tmp[i][j]=1-1.5*sum2/delta+0.5*(sum1/delta)**3
k_test_date3=tmp

y_pred=my_SVM.predict(k_test_date3[0:size_test])
score_SVM=accuracy_score(test_label1[0:size_test], y_pred)
print("The accuracy score of my kernel2-Spherical Kernel:", score_SVM)


#Log Kernel
n_dim=2025
d=2 #d是参数

my_SVM=SVC(kernel='precomputed')

tmp=np.empty((size_train,size_train),np.int)
for i in range(0,size_train):
    for j in range(0,size_train):
        sum1=0
        for k in range(0,n_dim):
            sum1=sum1+(train_data1[i][k]-train_data1[j][k])^d
        tmp[i][j]=-math.log(1+sum1)
k_train_data3=tmp

my_SVM.fit(k_train_data3[0:size_train], train_label1[0:size_train])

for i in range(0,size_test):
    for j in range(0,size_train):
        sum1=0
        for k in range(0,n_dim):
            sum1=sum1+(test_data1[i][k]-test_data1[j][k])^d
        tmp[i][j]=-math.log(1+sum1)
k_test_date3=tmp

y_pred=my_SVM.predict(k_test_date3[0:size_test])
score_SVM=accuracy_score(test_label1[0:size_test], y_pred)
print("The accuracy score of my kernel3-Log Kernel:", score_SVM) #log kernel 图像分割作用据说挺好的