# sklearn自带的SVM核的使用
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
#线性核
my_SVM=SVC(kernel='linear')
my_SVM.fit(train_data1[0:size_train], train_label1[0:size_train])
y_pred=my_SVM.predict(test_data1[0:size_test])
score_SVM=accuracy_score(test_label1[0:size_test], y_pred)
print("The accuracy score of Linear:", score_SVM)

#多项式核
my_SVM=SVC(kernel='poly')
my_SVM.fit(train_data1[0:size_train], train_label1[0:size_train])
y_pred=my_SVM.predict(test_data1[0:size_test])
score_SVM=accuracy_score(test_label1[0:size_test], y_pred)
print("The accuracy score of poly:", score_SVM)

#RBF核
my_SVM=SVC(kernel='rbf')
my_SVM.fit(train_data1[0:size_train], train_label1[0:size_train])
y_pred=my_SVM.predict(test_data1[0:size_test])
score_SVM=accuracy_score(test_label1[0:size_test], y_pred)
print("The accuracy score of rbf:", score_SVM)

#sigmoid核
my_SVM=SVC(kernel='sigmoid')
my_SVM.fit(train_data1[0:size_train], train_label1[0:size_train])
y_pred=my_SVM.predict(test_data1[0:size_test])
score_SVM=accuracy_score(test_label1[0:size_test], y_pred)
print("The accuracy score of sigmoid:", score_SVM)