from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
# from libsvm.python.svm import *
# from libsvm.python.svmutil import *
import numpy as np
from PIL import Image
import math
n_train = 60000 #The number of figures in training set
n_test = 10000  #The number of figures in testing set
fig_w = 45       #width of each figure

train_data = np.fromfile("mnist_train_data",dtype=np.uint8)
train_label = np.fromfile("mnist_train_label",dtype=np.uint8)

test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
test_label = np.fromfile("mnist_test_label",dtype=np.uint8)

train_data1 = list(train_data.reshape(n_train,fig_w*fig_w))
train_label1 = list(train_label)

test_data1 = list(test_data.reshape(n_test,fig_w*fig_w))
test_label1 = list(test_label)

from sklearn.decomposition import PCA
size_train=600
size_test=10000
print('a')
train_data5=[]
train_label5=[]
decomp_data=[]
for i in range(0,10):
    train_data5=[]
    for j in range(0,size_train):
        if(train_label1[j]==i):
            train_data5.append(train_data1[j])
            train_label5.append(train_label1[j])
    my_PCA=PCA(n_components=144) #1000为降维的大小
    tmp=my_PCA.fit_transform(train_data5)
    for k in range(0,len(tmp)):
        decomp_data.append(tmp[k])
    print(np.array(decomp_data).shape())
decomp_data=np.array(decomp_data).reshape(size_train,12,12)
plt.imshow(decomp_data[0])
plt.show()
print('a')
# test_data5=[]
# test_label5=[]
# decomp_test_data=[]
# for i in range(0,10):
#     test_data5=[]
#     for j in range(0,size_test):
#         if(train_label1[j]==i):
#             test_data5.append(test_data1[j])
#             test_label5.append(test_label1[j])
#     my_PCA=PCA(n_components=160) #1000为降维的大小
#     tmp=my_PCA.fit_transform(test_data5)
#     for k in range(0,len(tmp)):
#         decomp_test_data=np.concatenate((decomp_test_data,tmp[k]),axis=0)
# decomp_test_data=decomp_test_data.reshape(size_test,160)
#
# my_MLP=MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20,20,20), random_state=1)
# my_MLP.fit(decomp_data[0:size_train], train_label5[0:size_train])
# y_pred=my_MLP.predict(decomp_test_data[0:size_test])
# score_MLP=accuracy_score(test_label5[0:size_test], y_pred)
#
# print("The accuracy score of MLP with PCA:", score_MLP)