# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:44:04 2018

@author: SKCC16D00071
"""

import tensorflow as tf
import math

#x_train = [1,2,3]
#y_train = [1,2,3]
# Data
X = tf.placeholder(tf.float32, shape=[None,1])
Y = tf.placeholder(tf.float32, shape=[None,1])

# Variable to get
#W = tf.Variable(tf.random_normal([1]), name='weight')
W = tf.Variable(tf.random_normal([1,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
# hypothesis = X*W + b
hypothesis = tf.matmul(X,W) + b

# Cost function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimizer
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# for plotting
plot_x = []
plot_y = []

for step in range(4001):
    
    # Run together...
    #cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1,2,3,4,5], Y:[1,2,3,4,5]})
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], 
                                         feed_dict={X:[[1],[2],[3],[4],[5]], Y:[[1],[2],[3],[4],[5]]})
    
    plot_x.append(step)
    plot_y.append(math.log10(cost_val))
    if step % 200 == 0:
        print(step, cost_val, W_val, b_val)
        


import matplotlib.pylab as plt
plt.plot(plot_x, plot_y)
plt.show()