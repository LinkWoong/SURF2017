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

output_path = '/media/linkwong/File/CycleGAN'
batch_size = 1
pool_size = 50
sample_size = 10
max_images = 100

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



class Test():

    def input_load(self):

        directory_A = '/home/linkwong/SURF2017/acGAN-Implementation/mingrixiang.jpg'
        directory_B = '/home/linkwong/SURF2017/CycleGAN/patterned_leaves.jpg'

        filename_A = tf.train.match_filenames_once(directory_A)
        filename_B = tf.train.match_filenames_once(directory_B)

        self.queue_length_A = tf.size(filename_A)
        self.queue_length_B = tf.size(filename_B)

        init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])

        with tf.Session() as sess:

            sess.run(init)

            sess.run(filename_A)
            sess.run(filename_B)

            print sess.run(filename_A)
            print sess.run(filename_B)

        filename_queue_A = tf.train.string_input_producer(filename_A)
        filename_queue_B = tf.train.string_input_producer(filename_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [256, 256]), 127.5), 1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [256, 256]), 127.5), 1)


        # (256, 256, ?) with dynamic dimension
    def input_test(self):

        init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])

        with tf.Session() as sess:

            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            num_files_A = sess.run(self.queue_length_A)
            num_files_B = sess.run(self.queue_length_B)

            print num_files_A
            print num_files_B

            self.fake_image_A = np.zeros((pool_size, 1, img_height, img_width, img_layer))
            self.fake_image_B = np.zeros((pool_size, 1, img_height, img_width, img_layer))

            self.A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
            self.B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))

            for i in range(max_images):

                image_tensor = sess.run(self.image_A)
                if len(image_tensor) == img_size*batch_size*img_layer:

                    self.A_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))

            for i in range(max_images):

                image_tensor = sess.run(self.image_B)

                if  len(image_tensor) == img_size * batch_size * img_layer:

                    self.B_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layer))

            coord.request_stop()
            coord.join(threads)

    def preload(self):

        self.input_A = tf.placeholder(tf.float32, [batch_size, img_height. img_width, img_layer], name='input_A')
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_layer], name='input_B')

        self.fake_pool_A = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name='fake_pool_A')
        self.fake_pool_B = tf.placeholder(tf.float32, [None, img_height, img_width, img_layer], name='fake_pool_B')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.num_fake_inputs = 0
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')

        with tf.variable_scope('Model') as scope:

            self.fake_B = build_generator_resnet_9blocks(self.input_A, name='genA2B')
            self.fake_A = build_generaotr_resnet_9blocks(self.input_B, name='genB2A')

            self.judge_A = build_discriminator(self.input_A, name='disA')
            self.judge_B = build_discriminator(self.input_B, name='disB')

            scope.reuse_variables()

            self.judge_fake_A = build_discriminator(self.fake_A, name='disFA')
            self.judge_fake_B = build_discriminator(self.fake_B, name='disFB')
            self.cyclic_A = build_generator_resnet_9blocks(self.fake_B, name='cyclic_A')
            self.cyclic_B = build_generator_resnet_9blocks(self.fake_A, name='cyclic_B')

            scope.reuse_variables()

            self.judge_fake_A_pool = build_discriminator(self.judge_fake_A, name='poolFA')
            self.judge_fake_B_pool = build_discriminator(self.judge_fake_B, name='poolFB')


    def loss(self):















model = Test()
model.input_load()
model.input_test()
