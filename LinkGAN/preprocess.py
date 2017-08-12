import tensorflow as tf
import numpy as np
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from Essential import *
from keras.models import load_model
import pandas as pd 

mod = load_model('/media/linkwong/D/mod.h5')
pd.set_option('display.expand_frame_repr', False)

path = '/media/linkwong/D/1girl'
content_to_sketch = True

images = load_image(path)


for i in range(len(images)):
	temp = resize(images[i])
	name = '/media/linkwong/D/1girl/temp/resized_' + str(i) +'.jpeg'
	print name
	cv2.imwrite(name, temp)
new_path = '/media/linkwong/D/1girl/temp'

if content_to_sketch:
	images = load_image(new_path)
	

	for i in range(len(images)):

		width = float(images[i].shape[1])
		height = float(images[i].shape[0])
		depth = float(images[i].shape[2])
		print images[i].shape

		images[i] = images[i].transpose((2, 0, 1))

		light_map = np.zeros(images[i].shape, dtype=np.float)

		for j in range(3):
			light_map[j] = light_map_single(images[i][j])
		light_map = normalize_pic(light_map)
		light_map = resize_img_512_3d(light_map)

		line_mat = mod.predict(light_map, batch_size=1)
		line_mat = line_mat.transpose((3, 1, 2, 0))[0]
		line_mat = line_mat[0:int(height), 0:int(width), :]

		show_active_img_and_save('sketches', line_mat, '/media/linkwong/D' + str(i) + '.jpeg')
		line_mat = np.amax(line_mat, 2)
		show_active_img_and_save_denoise_filter2('sketches', line_mat, '/media/linkwong/D/1girl/temp/sketch/sketched_' + str(i) + '.jpeg')
		show_active_img_and_save_denoise_filter('sketches', line_mat, '/media/linkwong/D' + str(i) + '.jpeg')