from __future__ import print_function
from collections import defaultdict


import cPickle as pickle 
from PIL import Image
from six.moves import range
import matplotlib.pyplot as plt

import keras.backend as K 
from keras.datasets import mnist 
from keras.layers import Input, Dense, Reshape, Flatten, Embedding
from keras.layers import Dropout
from keras.layers import merge 
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D 
from keras.models import Sequential, Model 
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar 
from keras.utils import plot_model

import numpy as np 

np.random.seed(1337)

K.set_image_dim_ordering('tf')

def generator(latent_size):

	cnn = Sequential()

	cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
	cnn.add(Dense(128 * 7 * 7, activation='relu'))
	cnn.add(Reshape((128, 7, 7)))

	cnn.add(UpSampling2D(size=(2,2)))
	cnn.add(Conv2D(256, (5, 5), padding='same', activation='relu',kernel_initializer='glorot_normal'))

	cnn.add(UpSampling2D(size=(2,2)))
	cnn.add(Conv2D(128, (5, 5), padding='same', activation='relu',kernel_initializer='glorot_normal'))
	cnn.add(Conv2D(1, (2, 2), padding='same', activation='relu',kernel_initializer='glorot_normal'))
	latent = Input(shape=(latent_size, ))
	image_class = Input(shape=(1,), dtype='int32')

	cls = Flatten()(Embedding(10, latent_size, embeddings_initializer='glorot_normal')(image_class))

	h = merge([latent, cls], mode='mul')
	fake_image = cnn(h)

	return Model(outputs=fake_image, inputs=[latent, image_class])


def discriminator():

	cnn = Sequential()

	cnn.add(Conv2D(32, (3, 3), padding='same', strides=(2, 2), input_shape=(1, 28, 28)))
	cnn.add(LeakyReLU(alpha=0.2))
	cnn.add(Dropout(0.3))

	cnn.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
	cnn.add(LeakyReLU(alpha=0.2))
	cnn.add(Dropout(0.3))

	cnn.add(Conv2D(128, (3, 3), padding='same', strides=(2, 2)))
	cnn.add(LeakyReLU(alpha=0.2))
	cnn.add(Dropout(0.3))

	cnn.add(Conv2D(256, (3, 3), padding='same', strides=(1, 1)))
	cnn.add(LeakyReLU(alpha=0.2))
	cnn.add(Dropout(0.3))

	cnn.add(Flatten())

	image = Input(shape=(1, 28, 28))

	features = cnn(image)

	fake = Dense(1, activation='sigmoid', name='generation')(features)
	aux = Dense(10, activation='softmax', name='auxiliary')(features)

	return Model(outputs=[fake, aux],inputs=image)

if __name__ == '__main__':

	epochs = 50
	batch_size = 100
	latent_size = 100

	adam_lr = 0.0002
	adam_beta_1 = 0.5

	discriminator = discriminator()
	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
										loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

	generator = generator(latent_size)
	generator.compile(optimizer=Adam(lr=adam_lr,beta_1=adam_beta_1),
										loss='binary_crossentropy')

	latent = Input(shape=(latent_size,))
	image_class = Input(shape=(1,), dtype='int32')
	
	fake = generator([latent, image_class])

	discriminator.trainable = False
	fake, aux = discriminator(fake)
	