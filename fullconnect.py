import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
data_num = 60000 #The number of figures
data = np.fromfile("mnist_train_data",dtype=np.uint8)
label_pre = np.fromfile("mnist_train_label",dtype=np.uint8)

test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
test_label_pre = np.fromfile("mnist_test_label",dtype=np.uint8)

test_data = test_data.reshape(-1,45*45)
data = data.reshape(data_num,45*45)

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
xs=tf.placeholder(tf.float32,[None,45*45])
ys=tf.placeholder(tf.float32,[None,10])
keep_prob=tf.placeholder(tf.float32)

w1 = weight_variable([45 * 45, 512])
b1 = bias_variable([512])
h1 = tf.nn.relu(tf.matmul(xs / 255, w1) + b1)

w2 = weight_variable([512, 36])
b2 = bias_variable([36])
h2 = tf.nn.relu(tf.nn.dropout(tf.matmul(h1, w2) + b2, keep_prob))

w3 = weight_variable([36, 10])
b3 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h2, w3) + b3)

cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=1))
train_step=tf.train.AdadeltaOptimizer(1e-2).minimize(cross_entropy)

sess.run(tf.global_variables_initializer())
accuracy_test=[]
accuracy_train=[]
max_ac=0
for i in range(2600):
    for j in range(100):
        sess.run(train_step,
                 feed_dict={xs: data[600 * j:600 * (j + 1)], ys: label[600 * j:600 * (j + 1)], keep_prob: 0.5})
    if i % 10 == 0:
        ac_test = compute_accuracy(test_data, test_label)
        ac_train = compute_accuracy(data[0:1000], label[0:1000])
        print('step ', i)
        print('test ac=', ac_test)
        print('train ac=', ac_train)
        if ac_test > max_ac:
            max_ac = ac_test
        accuracy_test.append(ac_test)
        accuracy_train.append(ac_train)
print('the max accuracy on test set = ',max_ac)
x_label=[i*10 for i in range(1,261)]
plt.plot(x_label,accuracy_test,'r')
plt.plot(x_label,accuracy_train,'b')
plt.title('the performance of full_connect with dropout')
plt.xlabel('train step')
plt.xlabel('accuracy')
plt.show()

