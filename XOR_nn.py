# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:18:01 2018

@author: SKCC16D00071
"""

import tensorflow as tf

x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [[0],[1],[1],[0]]

# Input & Output
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Layer 1
W1 = tf.Variable(tf.random_normal([2,10]), name='weight1')
b1 = tf.Variable(tf.random_normal([10]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)

# Layer 2
W2 = tf.Variable(tf.random_normal([10,10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
layer2 = tf.sigmoid(tf.matmul(layer1, W2)+b2)

# Layer 3
W3 = tf.Variable(tf.random_normal([10,1]), name='weight3')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')

# Hypothesis
hypothesis = tf.sigmoid(tf.matmul(layer2, W3)+b3)

# Cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)* tf.log(1-hypothesis))

# Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# accuracy computation: True if hypotheisis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        
        if step % 2000 == 0:
            print(step, cost_val)
            
            # Accuracy Report
    h,c,a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("hypothesis : ", h, " Correct : ", c, " Accuracy: ", a)