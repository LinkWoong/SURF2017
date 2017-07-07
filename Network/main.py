import tensorflow as tf 
import os   
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import encoder
import vgg_model
import cv2
import numpy as np 
import inspect 


VGG = "/media/linkwong/D/imagenet-vgg-verydeep-19.mat"
path = "/home/linkwong/mingrixiang.jpg"

if os.path.exists(VGG):

	print 'VGG-19 Model is ready to use'

else:

	print "Failed to load VGG-19.mat"
    
input_image = cv2.imread(path)
input_image = cv2.resize(input_image, (256,256), interpolation=cv2.INTER_AREA).astype(np.float32)
if os.path.exists(path):

	print 'Image has been successfully loaded'
else:

	print 'Failed to load image'

#cv2.imshow('dick',input_image)
#cv2.waitKey() 
#cv2.destroyAllWindows() 
input_image[:,:,0] -= 103.939
input_image[:,:,1] -= 116.779
input_image[:,:,2] -= 123.68
input_image = np.expand_dims(np.transpose(input_image,(1,0,2)),axis=0)

print input_image.shape

input_image_ten = tf.Variable(initial_value=input_image, dtype=tf.float32, trainable=False)

with tf.Session() as sess:

	sess.run(tf.global_variables_initializer())
	net = vgg_model.vgg_model(VGG, input_image_ten)
