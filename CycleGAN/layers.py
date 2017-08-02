import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

batch_size = 1
pool_size = 50


def instance_norm(x):

    with tf.variable_scope("intance_norm"):

        epsilon = 1e-5
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

        scale = tf.get_variable('scale', [x.get_shape()[-1]], initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.02))

        offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer=tf.constant_initializer(0.0))

        output = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

        return output


def leakyrelu(x, leaky=0.2, name='leakyrelu'):

    with tf.variable_scope(name):

        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

    return f1 * x + f2 * abs(x)

def conv2d(inputconv, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding='SAME', name='conv2d', do_norm=True, do_relu=True, relufactor=0):

    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d(inputconv, o_d, f_w, s_w, padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev), biases_initializer=tf.constant_initializer(0.0))

        if do_norm:

            conv = instance_norm(conv)

        if do_relu:

            if (relufactor == 0):

                conv = tf.nn.relu(conv, 'relu')
            else:

                conv = leakyrelu(conv, relufactor, 'leakyrelu')

        return conv

def deconv2d(inputconv, output, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1, stddev=0.02, padding='VALID',name='deconv2d', do_norm=True, do_relu=True, relufactor=0):

    with tf.variable_scope(name):

        conv = tf.contrib.layers.conv2d_transpose(inputconv, o_d, [f_h, f_w], [s_h, s_w],padding, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(stddev=stddev), bias_initializer=tf.constant_initializer(0.0))

        if do_norm:

            conv = instance_norm(conv)

        if do_relu:

            if (relufactor == 0):

                conv = tf.nn.relu(conv, 'relu')

            else:

                conv = leakyrelu(conv, relufactor, 'leakyrelu')

        return conv

def build_resnet_block(inputconv, dim, name='resnet'):

    with tf.variable_scope(name):

        out_res = tf.pad(inputconv, [[0, 0]. [1, 1], [1, 1], [0, 0]], 'REFLECT')
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, 'VALID', 'c1')
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
        out_res = conv2d(out_res, dim, 3, 3, 1, 1,0.02, 'VALID', 'c2', do_relu=False)

    return tf.nn.relu(out_res + inputconv)



def build_generator_resnet_6blocks(inputconv, name='generator'):

    with tf.variable_scope(name):

        pad_input = tf.pad(inputconv, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
        o_c1 = conv2d(pad_input, 32, 7, 7, 1, 1, 0.02, name='c1')
        o_c2 = conv2d(o_c1, 32*2, 7, 7, 2, 2, 0.02, 'SAME', 'c2')
        o_c3 = conv2d(o_c2, 32*4, 7, 7, 2, 2, 0.02, 'SAME', 'c3')

        o_r1 = build_resnet_block(o_c3, 32*4, 'r1')
        o_r2 = build_resnet_block(o_r1, 32*4, 'r2')
        o_r3 = build_resnet_block(o_r2, 32*4, 'r3')
        o_r4 = build_resnet_block(o_r3, 32*4, 'r4')
        o_r5 = build_resnet_block(o_r4, 32*4, 'r4')
        o_r6 = build_resnet_block(o_r5, 32*4, 'r5')

        o_c4 = deconv2d(o_r6,










