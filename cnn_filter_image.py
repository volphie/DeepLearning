# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:29:08 2018

@author: SKCC16D00071
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # 현재 directory 기준으로

img = mnist.train.images[500].reshape(28,28)
plt.imshow(img, cmap='gray')
plt.show()

sess = tf.InteractiveSession()
img = img.reshape(-1,28,28,1)
W1 = tf.Variable(tf.random_normal([3,3,1,5], stddev=0.01))
conv2d = tf.nn.conv2d(img, W1, strides=[1,2,2,1], padding='VALID')
print(conv2d)
    
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
    
conv2d_img = np.swapaxes(conv2d_img,0,3)
    
print("Reshaping conved_img to ",conv2d_img.shape)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1)
    plt.imshow(one_img.reshape(13,13), cmap='gray')
    
plt.show()

sess.close()