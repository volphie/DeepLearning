# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 13:18:01 2018

@author: SKCC16D00071
"""

import tensorflow as tf


x_data = [[2,2], [2,3], [3,1], [4,3], [5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)

# Cost function
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)* tf.log(1-hypothesis))

# Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

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