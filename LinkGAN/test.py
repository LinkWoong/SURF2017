import tensorflow as tf
import numpy as np
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
import pandas as pd 

from keras.models import load_model
from preprocess import *
from essential import *
from network import *


path = '/media/linkwong/D/1girl'
mod_path = '/media/linkwong/D/mod.h5'

img_width = 512
img_height = 512
img_depth = 3

alpha = 0.3
beta = 0.9
batch_size = 1
pooling = 10
epochs = 100
global_step = 0
max_images = 3

content_to_sketch = True
log_device_placement = True
output_partition_graphs = True
start_train = True
start_test = True

content_path = '/media/linkwong/D/1girl/temp'
sketch_path = content_path + '/sketch'

if not os.path.exists(content_path):
	content_path, sketch_path = preprocess(path, mod_path, content_to_sketch)


style_path = '/media/linkwong/D/1girl/2294199.png'

def style_preprocess(style_path):

	temp = cv2.imread(style_path)
	dim = (512, 512)
	temp = cv2.resize(temp,(512, 512), interpolation=cv2.INTER_AREA)
	cv2.imwrite(style_path, temp)

	return style_path

style_preprocess(style_path)

class link():

	def setup(self, content_path, sketch_path, style_path, global_step):
		'''
		Arguments:

		content_path: the path of content images
		sketch_path: the path of sketch images
		style_path: the path of style image
		global_step: the global step
		'''

		content_match = tf.train.match_filenames_once(content_path + '/*.jpeg')
		#self.num_content = tf.size(content_match)

		sketch_match = tf.train.match_filenames_once(sketch_path + '/*.jpeg')
		#self.num_sketch = tf.size(sketch_match)

		style_match = tf.train.match_filenames_once(style_path)
		#self.num_style = tf.size(style_match)

		self.filename_content = tf.train.string_input_producer(content_match)
		self.filename_sketch = tf.train.string_input_producer(sketch_match)
		self.filename_style = tf.train.string_input_producer(style_match)

		reader = tf.WholeFileReader()

		_, content_image = reader.read(self.filename_content)
		_, sketch_image = reader.read(self.filename_sketch)
		_, style_image = reader.read(self.filename_style)

		#decode method depends on your image type

		self.content_read = tf.image.decode_jpeg(content_image)
		self.sketch_read = tf.image.decode_jpeg(sketch_image)
		self.style_read = tf.image.decode_png(style_image)

		self.content_input = np.zeros((max_images, img_width, img_height, img_depth))
		self.sketch_input = np.zeros((max_images, img_width, img_height, img_depth))
		self.style_input = np.zeros((1, img_width, img_height, img_depth))


		init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])

		with tf.Session() as sess:

			sess.run(init)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			#type: ndarray, shape: (512, 512, 3)

			#load each image into list
			for i in range(max_images):
				image_tensor = sess.run(self.content_read)
				self.content_input[i] = image_tensor.reshape((batch_size, img_width, img_height, img_depth))

			for i in range(max_images):
				image_tensor = sess.run(self.sketch_read)
				self.sketch_input[i] = image_tensor.reshape((batch_size, img_width, img_height, 1))

			for i in range(1):
				image_tensor = sess.run(self.style_read)
				self.style_input[i] = image_tensor.reshape((batch_size, img_width, img_height, img_depth))


			self.num_content = sess.run(tf.size(content_match))
			self.num_sketch = sess.run(tf.size(sketch_match))
			self.num_style = sess.run(tf.size(style_match))

			coord.request_stop()
			coord.join(threads)

		self.global_step = tf.Variable(global_step, dtype=tf.float32, trainable=False)

	def connect(self):

		self.content_conv_1 = resnet_9_layers(self.content_input)	
		


ass = link()
ass.setup(content_path, sketch_path, style_path, global_step)
ass.connect()

