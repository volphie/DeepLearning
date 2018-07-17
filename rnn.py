# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 13:37:02 2018

@author: SKCC16D00071
"""
import tensorflow as tf
import numpy as np

sample = ' I am working at SK'
idx2char = list(set(sample))
print(idx2char)

char2idx = {c:i for i, c in enumerate(idx2char)}
print(char2idx)

dic_size = len(char2idx) # one-hot encoding vector size
print(dic_size)

hidden_size = len(char2idx)
num_classes = len(char2idx)
batch_size = 1
sequence_length = len(sample) - 1
print(sequence_length) 

learning_rate = 0.1
sample_idx = [char2idx[c] for c in sample]
print(sample_idx)

x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

print("x_data", x_data)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
x_one_hot = tf.one_hot(X, num_classes)

cell= tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
         cell,
         x_one_hot,
         initial_state = initial_state,
         dtype = tf.float32)

X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
outputs = tf.reshape(outputs,[batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs, targets=Y, weights = weights)
cost = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
prediction = tf.argmax(outputs, axis=2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(50):
        l, _ = sess.run([cost, train], feed_dict = {X:x_data,Y:y_data})
        result = sess.run(prediction, feed_dict = {X:x_data})
        
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, "loss:", l, "Prediction", ''.join(result_str))
