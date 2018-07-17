# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:47:34 2018

@author: SKCC16D00071
"""

import tensorflow as tf

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],
          [1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]] # One-Hot Encoding

X = tf.placeholder(tf.float32, [None, 4])  # [i,j] i : No. of data, j : input dimension
Y = tf.placeholder(tf.float32, [None, 3])  # [i,j] i : No. of data, j : output dimension , No. of classification

nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight') # 행은 X의 입력 변수 Vector 크기와 같고, 열은 Output의 Vector의 Dimension과 같음
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1)) # Cross Entropy as a cost function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 400 ==0:
            print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
            
    
    # Testing
    a = sess.run(hypothesis, feed_dict={X:[[1,11,7,9]]})
    print(a, sess.run(tf.argmax(a,1)))
    
    b = sess.run(hypothesis, feed_dict={X:[[1,3,4,3]]})
    print(b, sess.run(tf.argmax(b,1)))