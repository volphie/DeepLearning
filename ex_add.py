# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 09:41:12 2018

@author: SKCC16D00071
"""

import tensorflow as tf

node1 = tf.constant(0, dtype=tf.int32)
node2 = tf.constant(5, dtype=tf.int32)
node3 = tf.add(node1, node2)

#sess = tf.Session()

# instead of line 14 and 20, don't need to close session. 
# When code gets out of the with block, the session object is deobjectized automatically
with tf.Session() as sess :
    print("Sum of node1 and node2 ", sess.run(node3))

#sess.close()