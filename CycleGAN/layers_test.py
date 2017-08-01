import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from layers import *

filename_queue = tf.train.string_input_producer(['/home/linkwong/SURF2017/acGAN-Implementation/mingrixiang.jpg'])

reader = tf.WholeFileReader()

key, value = reader.read(filename_queue)

img = tf.image.decode_jpeg(value)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    image = img.eval()

    print image.shape


    coord.request_stop()
    coord.join(threads)

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

features = 32

img_resize = tf.image.resize_images(img,(img_height, img_width), align_corners=False)

with tf.Session() as sess:

    print img_resize.shape

    img_resize.set_shape((256,256,3))

    o_c1 = conv2d(img_resize, 32, 7 ,7, 1, 1, 0.02, name='c1')
    o_c2 = conv2d(o_c1, 128, 3, 3, 2, 2, 0.02, name='c2')
    o_enc_A = conv2d(o_c2, 256, 3, 3, 2, 2, 0.02, name='c3')

    print o_c2.shape
    print o_enc_A.shape
