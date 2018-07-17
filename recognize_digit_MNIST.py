from __future__ import print_function

# MNIST 데이터를 사용할 때 항상 사용되는 문장
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True) # 현재 directory 기준으로

import tensorflow as tf
import time

t1 = time.time()

# Neural Network Core Model... Start here
num_steps = 5000
batch_size = 128
display_step = 100

num_input = 784
num_classes = 10

n_hidden_1 = 256
n_hidden_2 = 256
n_hidden_3 = 256
n_hidden_4 = 256

learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])

weights = {
        'h1' : tf.Variable(tf.random_normal([num_input,  n_hidden_1])),
        'h2' : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3' : tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4' : tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_hidden_4, num_classes]))
}

biases = {
        'b1' : tf.Variable(tf.random_normal([n_hidden_1])),        
        'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
        'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
        'b4' : tf.Variable(tf.random_normal([n_hidden_4])),
        'out': tf.Variable(tf.random_normal([num_classes])),
}

# Multi-Layer Perceptron
def mlp(x):
    L1 = tf.nn.relu(tf.matmul(x,  weights['h1']) + biases['b1'])
    L2 = tf.nn.relu(tf.matmul(L1, weights['h2']) + biases['b2'])
    L3 = tf.nn.relu(tf.matmul(L2, weights['h3']) + biases['b3'])
    L4 = tf.nn.relu(tf.matmul(L3, weights['h4']) + biases['b4'])
    Lout = tf.matmul(L4, weights['out'] ) + biases['out']
    return Lout

logits = mlp(X)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Neural Network Core Model... End here

# for comparison
prediction = tf.nn.softmax(logits) 
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    # Training
    for step in range(1, num_steps+1) :
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})
        if step % display_step == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={X:batch_x, Y:batch_y})
            print("step " + str(step) + ", Minibatch loss = " + "{:.4f}".format(loss) + ", Training Accuracy = "+"{:.4f}".format(acc*100) +"%")
    print("Optimization Finished!!")
    t2 = time.time()
    
    # Test
    print("Testing Accuracy : {:1f}%".format(sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})*100))
print("Learning Time: "+str(t2-t1)+" seconds")


