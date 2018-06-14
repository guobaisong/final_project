# "mnist_train_data" is the data file which contains a 60000*45*45 matrix(data_num*fig_w*fig_w)
# "mnist_train_label" is the label file which contains a 60000*1 matrix. Each element i is a number in [0,9].
# The dataset is saved as binary files and should be read by Byte. Here is an example of input the dataset and save a random figure.
print('a')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_num = 60000 #The number of figures
fig_w = 45       #width of each figure
print('a')
data = np.fromfile("mnist_train_data",dtype=np.uint8)
label_pre = np.fromfile("mnist_train_label",dtype=np.uint8)

test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
test_label_pre = np.fromfile("mnist_test_label",dtype=np.uint8)

#reshape the matrix
test_data = test_data.reshape(-1,45,45,1)
data = data.reshape(data_num,fig_w,fig_w,1)
sess = tf.Session()
test_label=sess.run(tf.one_hot(test_label_pre,depth=10))
label=sess.run(tf.one_hot(label_pre,depth=10))
def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    #strides [1,x_move,y_move,1]
    #padding='SAME'or'VALID'
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_3x3(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,3,3,1],padding='SAME')

xs=tf.placeholder(tf.float32,[None,45,45,1])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)
w1=weight_variable([4,4,1,32])
b1=bias_variable([32])
h1=tf.nn.relu(conv2d(xs/255,w1)+b1)
pool1=max_pool_3x3(h1)

w2=weight_variable([4,4,32,64])
b2=bias_variable([64])
h2=tf.nn.relu(conv2d(pool1,w2)+b2)
pool2=max_pool_3x3(h2)

w3=weight_variable([4,4,64,128])
b3=bias_variable([128])
h3=tf.nn.relu(conv2d(pool2,w3)+b3)

pool_conv=tf.reshape(h3,[-1,5*5*128])

W_fc1=weight_variable([5*5*128,1024])
b_fc1=bias_variable([1024])
h_fc1=tf.nn.relu(tf.nn.dropout(tf.matmul(pool_conv,W_fc1)+b_fc1,keep_prob))


W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

prediction=tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=1))

train_step=tf.train.AdadeltaOptimizer(1e-2).minimize(cross_entropy)
sess.run(tf.global_variables_initializer())

accuracy_test=[]
accuracy_train=[]
max_ac=0
for i in range(1000):
    for j in range(600):
        sess.run(train_step,feed_dict={xs:data[100*j:100*(j+1)],ys:label[100*j:100*(j+1)],keep_prob:0.5})
    if i%10==0:
        ac_test=compute_accuracy(test_data,test_label)
        ac_train=compute_accuracy(data[0:1000],label[0:1000])
        print('step ',i)
        print('test ac=',ac_test)
        print('train ac=',ac_train)
        if ac_test>max_ac:
            max_ac=ac_test
        accuracy_test.append(ac_test)
        accuracy_train.append(ac_train)
print('the max accuracy on test set = ',max_ac)
x_label=[i*10 for i in range(1,101)]
plt.plot(x_label,accuracy_test,'r')
plt.plot(x_label,accuracy_train,'b')
plt.title('the performance of CNN with dropout')
plt.xlabel('train step')
plt.xlabel('accuracy')
plt.show()



# print("After reshape:",data.shape)
#
# #choose a random index
# ind = np.random.randint(0,data_num)
#
# #print the index and label
# print("index: ",ind)
# print("label: ",label[ind])
#
# #save the figure
# im = Image.fromarray(data[ind])
# im.save("example.png")