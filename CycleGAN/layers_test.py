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

img_resize = tf.image.resize_images(img,[img_height, img_width], align_corners=False)

with tf.Session() as sess:

    print img_resize.shape
