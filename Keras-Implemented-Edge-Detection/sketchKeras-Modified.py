import cv2
from scipy import ndimage
import numpy as np 
import io
import os 
import pandas as pd 
from utilities import *
from keras.models import load_model

mod = load_model('/home/linkwong/Desktop/mod.h5')

pd.set_option('display.expand_frame_repr',False)
path = '/home/linkwong/mingrixiang.jpg'
path2 = '/home/linkwong/Desktop/backup.jpg'
image_initial = cv2.imread(path)


width = float(image_initial.shape[1])
height = float(image_initial.shape[0])
depth = float(image_initial.shape[2])

#cv2.imshow('dick',image_initial)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
new_width = 0
new_height = 0

if width > height :

	image_initial = cv2.resize(image_initial,(512, int(512/width * height)), interpolation=cv2.INTER_AREA)
	new_width = 512
	new_height = int(512 / width * height)
else:

	image_initial = cv2.resize(image_initial,(int(512/ width * height),512), interpolation=cv2.INTER_AREA)
	new_width = int(512 / width * height)
	new_height = 512
cv2.imshow('raw',image_initial)


image_initial = image_initial.transpose((2,0,1))
light_map = np.zeros(image_initial.shape, dtype=np.float)

for i in range(3):
	light_map[i] = light_map_single(image_initial[i])


light_map = normalize_pic(light_map)
light_map = resize_img_512_3d(light_map)

line_mat = mod.predict(light_map, batch_size=1)
line_mat = line_mat.transpose((3,1,2,0))[0]
line_mat = line_mat[0:int(new_height), 0:int(new_width),:]

show_active_img_and_save('sketchKeras_colored', line_mat,'sketchKeras_colored.jpg')
line_mat = np.amax(line_mat,2)
show_active_img_and_save_denoise_filter2('sketchKeras_enhanced',line_mat,'sketchKeras_enhanced.jpg')
show_active_img_and_save_denoise_filter('sketchKeras_pured',line_mat,'sketchKeras_pured.jpg')
cv2.waitKey(0)





