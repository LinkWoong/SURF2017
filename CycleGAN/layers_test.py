import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from layers import *
import random


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

        filename_queue_A = tf.train.string_input_producer(filename_A)
        filename_queue_B = tf.train.string_input_producer(filename_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A), [256, 256]), 127.5), 1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B), [256, 256]), 127.5), 1)


        # (256, 256, ?) with dynamic dimension
    def input_test(self,sess):

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

            if len(image_tensor) == img_size * batch_size * img_layer:

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



    def fake_image_pool(self, num_fakes, fake, fake_pool):


        if num_fakes < pool_size:

            fake_pool[num_fakes] = fake

            return fake
        else:
            if random.random() > 0.5:

                random_id = random.randint(0, pool_size - 1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake

                return temp
            else:

                return fake

    def loss(self):

        loss_dis_A = tf.reduce_mean(tf.squared_difference(self.judge_fake_A, 1))
        loss_dis_B = tf.reduce_mean(tf.squared_difference(self.judge_fake_B, 1))

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A - self.cyclic_A)) + tf.reduce_mean(tf.abs(self.input_B - self.cyclic_B))

        loss_gen_A = cyc_loss * 10 + loss_dis_B
        loss_gen_B = cyc_loss * 10 + loss_dis_A

        loss_dis_A = (tf.reduce_mean(tf.square(self.judge_fake_A_pool)) + tf.reduce_mean(tf.squared_difference(self.judge_fake_A, 1))) / 2.0
        loss_dis_B = (tf.reduce_mean(tf.square(self.judge_fake_B_pool)) + tf.reduce_mean(tf.squared_difference(self.judge_fake_B, 1))) / 2.0

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_variables = tf.trainable_variables()

        d_A = [i for i in self.model_variables if 'disA' in i.name]
        g_A = [i for i in self.model_variables if 'genA2B' in i.name]
        d_B = [i for i in self.model_variables if 'disB' in i.name]
        g_B = [i for i in self.model_variables if 'genB2A' in i.name]

        for i in self.model_variables:

            print i.name

    def save(self, sess, epoch):

        if not os.path.exists('/media/linkwong/File/CycleGAN/saves'):

            os.makedirs('/media/linkwong/File/CycleGAN/saves')
        for i in range(0, 10):

            fake_A_temp, fake_B_temp, cyclic_A_temp, cyclic_B_temp = sess.run([self.fake_A, self.fake_B, self.cyclic_A, self.cyclic_B], feed_dict={self.input_A:self.input_A[i], self.input_B:self.input_B[i]})
            imsave('/media/linkwong/File/CycleGAN/saves/fakeB_' + str(epoch) + '_' + str(i) + '.jpg', ((fake_A_temp[0] +1) * 127.5).astype(np.uint8))
            imsave('/media/linkwong/File/CycleGAN/saves/fakeA_' + str(epoch) + '_' + str(i) + '.jpg', ((fake_B_temp[0] +1) * 127.5).astype(np.uint8))
            imsave('/media/linkwong/File/CycleGAN/saves/cycA_' + str(epoch) + '_' + str(i) + '.jpg', ((cyclic_A_temp[0] +1) * 127.5).astype(np.uint8))
            imsave('/media/linkwong/File/CycleGAN/saves/cycB_' + str(epoch) + '_' + str(i) + '.jpg',((cyclic_B_temp[0] +1) * 127.5).astype(np.uint8))
            imsave('/media/linkwong/File/CycleGAN/saves/inputA_' + str(epoch) + '_' + str(i) + '.jpg', ((self.input_A[i][0] +1) * 127.5).astype(np.uint8))
            imsave('/media/linkwong/File/CycleGAN/saves/inputB_' + str(epoch) + '_' + str(i) + '.jpg', ((self.input_B[i][0] +1) * 127.5).astype(np.uint8))


    def train_model(self):

        self.input_load()
        self.preload()

        self.loss()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.

model = Test()
model.input_load()
model.input_test()
