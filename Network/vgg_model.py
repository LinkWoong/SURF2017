import tensorflow as tf 
import scipy.io
import numpy as np 
import cv2
import os   
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

path = '.'
path_image = '.'
input_image = cv2.imread(path_image)

def weights(vgg_layer, layer, expected_layer):

    #load trained weights and bias from VGG_19

	W = vgg_layer[0][layer][0][0][2][0][0]
	b = vgg_layer[0][layer][0][0][2][0][1]

	layer_name = vgg_layer[0][layer][0][0][0][0]
	assert layer_name = expected_layer

	return W, b.reshape(b.size)

def conv2d_relu(vgg, pre, layer, layer_name): 

	# convolution 2d with Relu as activation function
 	with tf.variable_scope(layer_name) as scope:

		W, b = weights(vgg, layer, layer_name)
		W = tf.constant(W, dtype=None, name='weights')
		b = tf.constant(b, dtype=None, name='bias')
		conv2d = tf.nn.conv2d(pre, filter=W, strides=[1,1,1,1], padding='SAME')

	conv2d_activated = tf.nn.relu(conv2d+b)

	return conv2d_activated

# I choose average pooling method
 
def average_pooling_layer(pre):

	return tf.nn.avg_pool(pre, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def maximum_pooling_layer(pre):

	return tf.nn.max_pool(pre, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#load VGG model

def vgg_model(path, input_image):
    
	vgg = scipy.io.loadmat(path)
	vgg_layer = vgg['layers']
     
    #dict is used instead of class
    #five convolution layers 
	net={}

	net['conv1_1'] = conv2d_relu(vgg_layer, input_image, 0, 'conv1_1')
	net['conv1_2'] = conv2d_relu(vgg_layer, net['conv1_1'], 2, 'conv1_2')
	net['conv1_3'] = conv2d_relu(vgg_layer, net['conv1_2'], 4, 'conv1_3')
	net['avg_pool1'] = average_pooling_layer(net['conv1_3'])

	net['conv2_1'] = conv2d_relu(vgg_layer, net['avg_pool1'], 7, 'conv2_1')
	net['conv2_2'] = conv2d_relu(vgg_layer, net['conv2_1'], 9, 'conv2_2')
	net['conv2_3'] = conv2d_relu(vgg_layer, net['conv2_2'], 11, 'conv2_3')
	net['avg_pool2'] = average_pooling_layer(net['conv2_2'])

	net['conv3_1'] = conv2d_relu(vgg_layer, net['avg_pool2'], 14, 'conv3_1')
	net['conv3_2'] = conv2d_relu(vgg_layer, net['conv3_1'], 16, 'conv3_2')
	net['conv3_3'] = conv2d_relu(vgg_layer, net['conv3_2'], 18, 'conv3_3')
	net['avg_pool3'] = average_pooling_layer(net['conv3_3'])

	net['conv4_1'] = conv2d_relu(vgg_layer, net['avg_pool3'], 21, 'conv4_1')
	net['conv4_2'] = conv2d_relu(vgg_layer, net['conv4_1'], 23, 'conv4_2')
	net['conv4_3'] = conv2d_relu(vgg_layer, net['conv4_2'], 25, 'conv4_3')
	net['avg_pool4'] = average_pooling_layer(net['conv4_3'])

	net['conv5_1'] = conv2d_relu(vgg_layer, net['avg_pool4'], 28, 'conv5_1')
	net['conv5_2'] = conv2d_relu(vgg_layer, net['conv5_1'], 30, 'conv5_2')
	net['conv5_3'] = conv2d_relu(vgg_layer, net['conv5_2'], 32, 'conv5_3')
	net['avg_pool5'] = average_pooling_layer(net['conv5_3'])

	return net 

#fully convolution layer with softmax output

def fc_layer(vgg, net, layer, layer_name):

	W, b = weights(vgg, layer, layer_name)
	W = tf.constant(W, dtype=None, name='weights')
	b = tf.constant(b ,dtype=None, name='bias')

    fc = tf.nn.bias_add(tf.matmul(net['avg_pool5'], W) + b)
    fc_softmax = tf.nn.softmax(fc, name='softmax')

    return fc_softmax
