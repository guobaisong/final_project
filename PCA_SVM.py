# PCA加SVM对MNIST数据进行分类
# 课程：Machine Learning CS420 
# By: 万俊成

import pandas as pd
from sklearn.neural_network import MLPClassifier  
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
from sklearn.decomposition import PCA  
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
#注意这里10 100 10 是可以调参数，的以便于跑出结果
for n_components in range(10,100,10):
    train_data5=[]
    train_label5=[]
    decomp_data=[]

    decomp_test_data=[]

    #下面是创建10个降维矩阵，同时对训练数据降维，由于python中的对象不仅仅是矩阵，而是一个特殊的对象，所以我只好重复代码
    #注意python是引用传递，所以必须创建多个my_PCA，我之前就是在这里出了巨大的bug，调试了很久
    i=0
    train_data5=[]
    my_PCA0=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])  
    pca0=my_PCA0.fit(train_data5)
    tmp_train=pca0.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)  

    i=1
    train_data5=[]
    my_PCA1=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])           
    pca1=my_PCA1.fit(train_data5)
    tmp_train=pca1.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    i=2
    train_data5=[]
    my_PCA2=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])          
    pca2=my_PCA2.fit(train_data5)
    tmp_train=pca2.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    i=3
    train_data5=[]
    my_PCA3=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])          
    pca3=my_PCA3.fit(train_data5)
    tmp_train=pca3.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    i=4
    train_data5=[]
    my_PCA4=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])           
    pca4=my_PCA4.fit(train_data5)
    tmp_train=pca4.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    i=5
    train_data5=[]
    my_PCA5=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])          
    pca5=my_PCA5.fit(train_data5)
    tmp_train=pca5.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    i=6
    train_data5=[]
    my_PCA6=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])           
    pca6=my_PCA6.fit(train_data5)
    tmp_train=pca6.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    i=7
    train_data5=[]
    my_PCA7=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])          
    pca7=my_PCA7.fit(train_data5)
    tmp_train=pca7.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    i=8
    train_data5=[]
    my_PCA8=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])           
    pca8=my_PCA8.fit(train_data5)
    tmp_train=pca8.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    i=9
    train_data5=[]
    my_PCA9=PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])        
    pca9=my_PCA9.fit(train_data5)
    tmp_train=pca9.transform(train_data5)
    for k in range(0,len(tmp_train)):      
        decomp_data=np.concatenate((decomp_data,tmp_train[k]),axis=0)

    decomp_data=decomp_data.reshape(size_train,n_components)
	
#创建SVM分类器，注意这里poly改为rbf可以尝试rbf的分类效果  
    score_SVM=[0 for i in range(0,10)]
    my_SVM=SVC(kernel='poly')
    my_SVM.fit(decomp_data[0:size_train], train_label5[0:size_train])
	
#对测试数据进行降维
    n_hit=0
    for i in range(0,size_test):  
        #分别尝试10个降维矩阵
        decomp_test_data_0=pca0.transform([test_data1[i]])
        decomp_test_data_1=pca1.transform([test_data1[i]])
        decomp_test_data_2=pca2.transform([test_data1[i]])
        decomp_test_data_3=pca3.transform([test_data1[i]])
        decomp_test_data_4=pca4.transform([test_data1[i]])
        decomp_test_data_5=pca5.transform([test_data1[i]])
        decomp_test_data_6=pca6.transform([test_data1[i]])
        decomp_test_data_7=pca7.transform([test_data1[i]])
        decomp_test_data_8=pca8.transform([test_data1[i]])
        decomp_test_data_9=pca9.transform([test_data1[i]])

        y_pred_0=my_SVM.predict(decomp_test_data_0)
        score_SVM[0]=accuracy_score([test_label1[i]], y_pred_0)

        y_pred_1=my_SVM.predict(decomp_test_data_1)
        score_SVM[1]=accuracy_score([test_label1[i]], y_pred_1)

        y_pred_2=my_SVM.predict(decomp_test_data_2)
        score_SVM[2]=accuracy_score([test_label1[i]], y_pred_2)

        y_pred_3=my_SVM.predict(decomp_test_data_3)
        score_SVM[3]=accuracy_score([test_label1[i]], y_pred_3)

        y_pred_4=my_SVM.predict(decomp_test_data_4)
        score_SVM[4]=accuracy_score([test_label1[i]], y_pred_4)

        y_pred_5=my_SVM.predict(decomp_test_data_5)
        score_SVM[5]=accuracy_score([test_label1[i]], y_pred_5)

        y_pred_6=my_SVM.predict(decomp_test_data_6)
        score_SVM[6]=accuracy_score([test_label1[i]], y_pred_6)

        y_pred_7=my_SVM.predict(decomp_test_data_7)
        score_SVM[7]=accuracy_score([test_label1[i]], y_pred_7)

        y_pred_8=my_SVM.predict(decomp_test_data_8)
        score_SVM[8]=accuracy_score([test_label1[i]], y_pred_8)

        y_pred_9=my_SVM.predict(decomp_test_data_9)
        score_SVM[9]=accuracy_score([test_label1[i]], y_pred_9)
        #判断是否预测正确
        if score_SVM[test_label1[i]]==1:
            n_hit=n_hit+1
    
    print("-----------------------------------------------")
    print("Number of components:",n_components)
    print("The accuracy score of PCA+SVM(poly kernel):", n_hit/size_test)