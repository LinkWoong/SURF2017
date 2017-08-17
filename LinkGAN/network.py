import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

img_width = 512
img_height = 512
img_depth = 3

filter_width = 7
filter_height = 7
stride_width = 1
stride_height = 1


def instance_norm(x):

	epsilon = 1e-5
	mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
	tf.get_variable_scope().reuse = True
	scale = tf.get_variable('scale2', [x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))
	offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))

	dive = tf.cast(tf.div(x - mean, tf.sqrt(var + epsilon)), tf.float32)

	print dive.dtype

	output = scale * dive  + offset

	return output

def leakyrelu(x, leaky=0.2):

	f1 = 0.5 * (1 + leaky)
	f2 = 0.5 * (1 - leaky)

	return f1 * x + f2 * abs(x)

def conv2d(inputconv, output_dim, filter_width, filter_height, stride_width, stride_height, stddev, padding='VALID', do_norm=True, do_relu=True, relufac=0):

	conv = tf.contrib.layers.conv2d(inputconv, output_dim, filter_width, stride_width, padding, activation_fn=None, 
		weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=tf.constant_initializer(0.0))

	if do_norm:

		conv = instance_norm(conv)
	if do_relu:

		if (relufac == 0):
			conv = tf.nn.relu(conv)
		else:
			conv = leakyrelu(conv, relufac)

	return conv

def deconv2d(inputconv, output_dim, filter_width, filter_height, stride_width, stride_height, stddev, padding='VALID', do_norm=True, do_relu=True, relufac=0):

	deconv = tf.contrib.layers.conv2d_transpose(inputconv, output_dim, [filter_height, filter_width], [stride_height, stride_width], padding, activation_fn=None,
		weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=tf.constant_intializer(0.0))

	if do_norm:

		deconv = instance_norm(deconv)
	if do_relu:

		if (relufac == 0):
			deconv = tf.nn.relu(deconv)
		else:
			deconv = leakyrelu(deconv, relufac)

	return deconv



def resnet(inputconv, dim):

	inputconv_pad_1 = tf.pad(inputconv,[[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
	inputconv_conv = conv2d(inputconv_pad, dim, 3, 3, 1, 1, 0.02, 'VALID')
	inputconv_pad_2 = tf.pad(inputconv_conv, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
	output = conv2d(inputconv_pad_2, 3, 3, 1, 1, 0.02, 'VALID', do_relu=False)

	return tf.nn.relu(output + inputconv)

def resnet_6_layers(inputconv):

	inputconv_pad = tf.pad(inputconv, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
	conv_1 = conv2d(inputconv_pad, 32, 7 ,7, 1, 1, 0.02)
	conv_2 = conv2d(conv_1, 32*2, 7, 7, 2, 2, 0.02, 'SAME')
	conv_3 = conv2d(conv_2, 32*4, 7, 7, 2, 2, 0.02, 'SAME')

	res_1 = resnet(conv_3, 32*4)
	res_2 = resnet(res_1, 32*4)
	res_3 = resnet(res_2, 32*4)
	res_4 = resnet(res_3, 32*4)
	res_5 = resnet(res_4, 32*4)
	res_6 = resnet(res_5, 32*4)


	deconv_1 = deconv2d(res_6, 32*2, 7, 7, 2, 2, 0.02, 'SAME')
	deconv_2 = deconv2d(deconv_1, 32, 7, 7, 2, 2, 0.02, 'SAME')
	deconv_pad = tf.pad(deconv_2, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
	out = conv2d(deconv_2, img_depth, 7, 7, 1, 1, 0.02, 'SAME', do_relu=False)

	output = tf.nn.tanh(out)

	return output

def resnet_9_layers(inputconv):

	inputconv_pad = tf.pad(inputconv, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
	conv_1 = conv2d(inputconv_pad, 32, 7 ,7, 1, 1, 0.02)
	conv_2 = conv2d(conv_1, 32*2, 7, 7, 2, 2, 0.02, 'SAME')
	conv_3 = conv2d(conv_2, 32*4, 7, 7, 2, 2, 0.02, 'SAME')

	res_1 = resnet(conv_3, 32*4)
	res_2 = resnet(res_1, 32*4)
	res_3 = resnet(res_2, 32*4)
	res_4 = resnet(res_3, 32*4)
	res_5 = resnet(res_4, 32*4)
	res_6 = resnet(res_5, 32*4)
	res_7 = resnet(res_6, 32*4)
	res_8 = resnet(res_7, 32*4)
	res_9 = resnet(res_8, 32*4)

	deconv_1 = deconv2d(res_9, 32*2, 7, 7, 2, 2, 0.02, 'SAME')
	deconv_2 = deconv2d(deconv_1, 32, 7, 7, 2, 2, 0.02, 'SAME')
	#deconv_pad = tf.pad(deconv_2, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
	out = conv2d(deconv_2, img_depth, 7, 7, 1, 1, 0.02, 'SAME', do_relu=False)

	output = tf.nn.tanh(out)

	return output

def discriminator(inputconv):

	conv_1 = conv2d(inputconv, 64, 4, 4, 2, 2, 0.02, 'SAME', do_norm=False, relufac=0.2)
	conv_2 = conv2d(conv_1, 64*2, 4, 4, 2, 2, 0.02, 'SAME', relufac=0.2)
	conv_3 = conv2d(conv_2, 64*4, 4, 4, 2, 2, 0.02, 'SAME', relufac=0.2)
	conv_4 = conv2d(conv_3, 64*8, 4, 4, 2, 2, 0.02, 'SAME', relufac=0.2)
	conv_5 = conv2d(conv_4, 1, 4, 4, 2, 2, 0.02, 'SAME', relufac=0.2)

	return conv_5
	