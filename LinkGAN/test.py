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
		filename_content = tf.train.string_input_producer(tf.train.match_filenames_once(content_path + '/*.jpeg'))
		filename_sketch = tf.train.string_input_producer(tf.train.match_filenames_once(sketch_path + '/*.jpeg'))

		reader = tf.WholeFileReader()
		
