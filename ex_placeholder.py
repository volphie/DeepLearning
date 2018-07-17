# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 10:02:06 2018

@author: SKCC16D00071
"""

import tensorflow as tf

a = tf.placeholder(tf.int32, shape=(3,1))
b = tf.placeholder(tf.int32, shape=(1,3))
c = tf.matmul(a,b)

with tf.Session() as sess:
    print("a x b : \n", sess.run(c, feed_dict={a:[[3],[2],[1]], b:[[1,2,3]]}))