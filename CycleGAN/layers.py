import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

def leakyrelu(x, leaky=0.2, name='leakyrelu'):

    with tf.variable_scope(name):

        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)

def conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding='VALID', name='conv2d', do_norm=True, do_relu=True, relufactor=0):

    witht tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(inputconv, o_d, f_w, s_w, padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=tf.constant_initializer(0.0))

        if do_norm:

            conv = instance_norm(conv)

        if do_relu:

            if (relufactor == 0):

                conv = tf.nn.relu(conv, 'relu')
            else:

                conv = leakyrelu(conv, relufactor, 'leakyrelu')

        return conv


