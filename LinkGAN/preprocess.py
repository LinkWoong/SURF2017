import tensorflow as tf
import numpy as np
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
from essential import *
from keras.models import load_model
import pandas as pd 

#This function transforms the colored image into sketch, the shape is (512, 512, 3)


def preprocess(image_path, mod_path, content_to_sketch):

	mod = load_model(mod_path)
	pd.set_option('display.expand_frame_repr', False)

	images = load_image(image_path)
	new_path = image_path + '/temp'

	for i in range(len(images)):

		temp = resize(images[i])
		if not os.path.exists(new_path):
			os.makedirs(new_path)
		name = new_path + '/resized_' + str(i) + '.jpeg'
		print name

		cv2.imwrite(name, temp)

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
			

			#show_active_img_and_save('sketches', line_mat, './' + str(i) + '.jpeg')
			line_mat = np.amax(line_mat, 2)
			


			show_active_img_and_save_denoise_filter2('sketches', line_mat, new_path, i)
			

			sketch_path = new_path + '/sketch'
			content_path = new_path
	else:

		content_path = new_path
		sketch_path = None

	return content_path, sketch_path