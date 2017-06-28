import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 30


mnist = input_data.read_data_sets('/home/linkwong/desktop2/surf/cs20si/tf-stanford-tutorials/data/mnist', one_hot=True) 


X = tf.placeholder(tf.float32, [batch_size, 784], name='image')
Y = tf.placeholder(tf.float32, [batch_size, 10], name='label')



w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 10]), name="bias")

logits = tf.matmul(X, w) + b 



def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#Convolution layer 1

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

X_image = tf.reshape(X,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(X_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#Convolution layer 2

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#FC layer

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

entropy = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('/home/linkwong/desktop2/Surf', sess.graph)
	
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			# TO-DO: run optimizer + fetch loss_batch
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
			# 
			total_loss += loss_batch
		print 'Average loss epoch {0}: {1}'.format(i, total_loss/n_batches)

	print 'Total time: {0} seconds'.format(time.time() - start_time)

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	# test the model
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		_, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch}) 
		preds = tf.nn.softmax(logits_batch)
		correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
		accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
		total_correct_preds += sess.run(accuracy)	
	
	print 'Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples)
	writer.close()
