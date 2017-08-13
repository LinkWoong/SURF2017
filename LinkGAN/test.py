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

content_to_sketch = True
log_device_placement = True
output_partition_graphs = True
start_train = True
start_test = True

content_path, sketch_path = preprocess(path, mod_path, content_to_sketch)

print content_path
print sketch_path

style_path = '/media/linkwong/D/1girl/2294199.png'


class link():

	def setup(self, content_path, sketch_path, style_path, global_step):

		self.global_step = global_step

		content_match = tf.train.match_filenames_once(content_path + '/*.jpeg')
		self.num_content = tf.size(content_match)

		sketch_match = tf.train.match_filenames_once(sketch_path + '/*.jpeg')
		self.num_sketch = tf.size(sketch_match)

		style_match = tf.train.match_filenames_once(style_path)
		self.num_style = tf.size(style_match)

		self.filename_content = tf.train.string_input_producer(content_match)
		self.filename_sketch = tf.train.string_input_producer(sketch_match)
		self.filename_style = tf.train.string_input_producer(style_match)

		reader = tf.WholeFileReader()

		_, content_image = reader.read(self.filename_content)
		_, sketch_image = reader.read(self.filename_sketch)
		_, style_image = reader.read(self.filename_style)

		#decode method depends on your image type

		content_tensor = tf.image.decode_jpeg(content_image)
		sketch_tensor = tf.image.decode_jpeg(sketch_image)
		style_tensor = tf.image.decode_png(style_image) 


		init = ([tf.global_variables_initializer(), tf.local_variables_initializer()])

		with tf.Session() as sess:

			sess.run(init)

			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)

			self.content = sess.run(content_tensor)
			self.sketch = sess.run(sketch_tensor)
			self.style = sess.run(style_tensor)

			#print num_content, num_sketch, num_style
			
			coord.request_stop()
			coord.join(threads)


ass = link()
ass.setup(content_path, sketch_path, style_path, global_step)

