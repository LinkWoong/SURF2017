import os   
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import encoder
import cv2
import numpy as np 
import VGG_Keras
import keras

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

print keras.__version__
VGG = "/media/linkwong/D/vgg19_weights.h5"
path = "/home/linkwong/mingrixiang.jpg"

if os.path.exists(VGG):

	print 'VGG-19 Model is ready to use'

else:

	print "Failed to load VGG-19.mat"
    
input_image = cv2.imread(path)

if os.path.exists(path):

	print 'Image has been successfully loaded'
else:

	print 'Failed to load image'

input_image = cv2.resize(input_image, (256,256)).astype(np.float32)


#cv2.imshow('dick',input_image)
#cv2.waitKey() 
#cv2.destroyAllWindows() 
input_image[:,:,0] -= 103.939
input_image[:,:,1] -= 116.779
input_image[:,:,2] -= 123.68

input_image = input_image.transpose((2,0,1))
input_image = np.expand_dims(input_image, axis=0)

print input_image.shape

model = VGG_Keras.VGG_19('/media/linkwong/D/vgg19_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy')
out = model.predict(input_image)

print np.argmax(out)