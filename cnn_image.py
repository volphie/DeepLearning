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

img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap='gray')
plt.show()