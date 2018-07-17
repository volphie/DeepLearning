# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:11:55 2018

@author: SKCC16D00071
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # 현재 directory 기준으로

learning_rate = 0.001
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32, [None, 784])
# -1 : 크기를 모른다. 훈련할 이미지의 갯수
# 28 : X
# 28 : Y
#  1 : color map , 흑백을 의미
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

#Layer 1
# 3x3, 1 image 32 개 feature를 갖는 filter
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))

# padding ='SAME' : 이미지를 크기를 같이하는 padding
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
# strides는 kernel size와 같게
# ksize가 가장 중요한 값 pooling에 의한 downsizing을 정의하는 것
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Layer2
# 3x3, 32개 image를 받아서 64 개 feature를 갖는 filter
W2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))

# padding ='SAME' : 이미지를 크기를 같이하는 padding
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
# strides는 ksize와 같게
# ksize가 가장 중요한 값 pooling에 의한 downsizing을 정의하는 것
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# Vectorization for classification layer input
L2_flat = tf.reshape(L2, [-1, 7*7*64])

# Layer 3 : Classification Layer, 1 Layer Perceptron with 10 classification
#W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer = tf.contrib.layers.xavier_initializer())
W3 =  tf.Variable(tf.random_normal([7*7*64, 10]))
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("Learning started....")
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict={X:batch_xs, Y:batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost+= c/total_batch
    
    print('Epoch:', '%04d'%(epoch+1), 'cost = ', '{:9f}'.format(avg_cost))
    
print('Learning Finished')

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:',sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))










