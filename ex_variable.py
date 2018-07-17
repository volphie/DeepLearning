# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:56:30 2018

@author: SKCC16D00071
"""

import tensorflow as tf

node = tf.Variable(tf.zeros([2,2])) # 2 by 2 matrix with 

with tf.Session() as sess :
    
    sess.run(tf.global_variables_initializer())
    
    print("Tensor value before addition:\n", sess.run(node))