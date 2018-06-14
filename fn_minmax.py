import tensorflow as tf
import numpy as np
import random as rd
import matplotlib.pyplot as plt

data_num = 60000 #The number of figures
fig_w = 45       #width of each figure

data = np.fromfile("mnist_train_data",dtype=np.uint8)
label_pre = np.fromfile("mnist_train_label",dtype=np.uint8)

test_data = np.fromfile("mnist_test_data",dtype=np.uint8)
test_label_pre = np.fromfile("mnist_test_label",dtype=np.uint8)

test_data = test_data.reshape(-1,45*45)
data = data.reshape(data_num,fig_w*fig_w)
#reshape the matrix
sess = tf.Session()

train_set=[[] for i in range(10)]
train_set_size=[0 for i in range(10)]
for i in range(len(data)):
    train_set[int(label_pre[i])].append(data[i])
    train_set_size[int(label_pre[i])]+=1

test_set=[[] for i in range(10)]
test_set_size=[0 for i in range(10)]
for i in range(len(test_data)):
    test_set[int(test_label_pre[i])].append(test_data[i])
    test_set_size[int(test_label_pre[i])]+=1

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre=sess.run(prediction,feed_dict={xs:v_xs})
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
ys=tf.placeholder(tf.float32,[None,1])
keep_prob=tf.placeholder(tf.float32)
correct_matrix=[[0 for j in range(10)] for i in range(10)]
for i in range(10):
    for j in range(i+1,10):
        if j==i:
            continue
        train_data=[]
        train_label=[]
        prediction_data=[]
        prediction_label=[]
        seti_count=0
        setj_count=0
        for k in range(len(train_set[i])+len(train_set[j])):
            r=rd.randint(0,1)
            if seti_count!=train_set_size[i] and (r==1 or setj_count==train_set_size[j]):
                train_data.append(train_set[i][seti_count])
                train_label.append([1.])
                seti_count+=1
            if setj_count!=train_set_size[j] and (r==0 or seti_count==train_set_size[i]):
                train_data.append(train_set[j][setj_count])
                train_label.append([0.])
                setj_count+=1
        seti_count = 0
        setj_count = 0
        for k in range(len(test_set[i])+len(test_set[j])):
            r=rd.randint(0,1)
            if seti_count!=test_set_size[i] and (r==1 or setj_count==test_set_size[j]):
                prediction_data.append(test_set[i][seti_count])
                prediction_label.append([1.])
                seti_count+=1
            if setj_count!=test_set_size[j] and (r==0 or seti_count==test_set_size[i]):
                prediction_data.append(test_set[j][setj_count])
                prediction_label.append([0.])
                setj_count+=1
        total_size=train_set_size[i]+train_set_size[j]
        w1=weight_variable([45*45,512])
        b1=bias_variable([512])
        h1=tf.nn.relu(tf.matmul(xs/255,w1)+b1)

        w2=weight_variable([512,36])
        b2=bias_variable([36])
        h2=tf.nn.relu(tf.nn.dropout(tf.matmul(h1,w2)+b2,keep_prob))

        w3=weight_variable([36,1])
        b3=bias_variable([1])
        h3=tf.nn.sigmoid(tf.matmul(h2,w3)+b3)

        h3_bar=1-h3

        cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(h3)+(1-ys)*tf.log(1-h3),reduction_indices=1))

        train_step=tf.train.AdadeltaOptimizer(1e-2).minimize(cross_entropy)

        sess=tf.Session()
        sess.run(tf.global_variables_initializer())
        max_ac=0
        for it in range(600):
            for step in range(100):
                sess.run(train_step,feed_dict={xs:train_data[(step*int(total_size/100)):((step+1)*int(total_size/100))],
                                               ys:train_label[(step*int(total_size/100)):((step+1)*int(total_size/100))],
                                               keep_prob:0.5})
            if it%10==0:
                y_pre = sess.run(h3, feed_dict={xs: prediction_data,keep_prob:1})
                correct_rate=sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.reduce_sum(tf.round(y_pre),reduction_indices=1),
                                                               tf.reduce_sum(prediction_label,reduction_indices=1)),
                                                      tf.float32)))
                y_test=sess.run(h3,feed_dict={xs:test_data,keep_prob:1})
                if correct_rate>max_ac:
                    max_ac=correct_rate
                    np.save('test_acuuracy_for_'+str(i)+'_to_'+str(j),np.array(sess.run(tf.reduce_sum(y_test,reduction_indices=1))))
                    np.save('test_acuuracy_for_'+str(j)+'_to_'+str(i),np.array(sess.run(tf.reduce_sum(1-y_test,reduction_indices=1))))
                    print(str(i)+'_and_'+str(j)+' save successfully')
                # print('test ac=',correct_rate)
                # y_pre = sess.run(tf.round(h3), feed_dict={xs: train_data[0:1000], keep_prob: 1})
                # correct_rate =sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.reduce_sum(y_pre, reduction_indices=1),
                #                                                tf.reduce_sum(train_label[0:1000], reduction_indices=1)),
                #                                       tf.float32)))
                # print('train_ac=',correct_rate)
        correct_matrix[i][j]=max_ac
        correct_matrix[j][i]=max_ac
        print(i,j,max_ac)
for i in range(10):
    for j in range(10):
        print(correct_matrix[i][j],' ',end='')
    print('')