from __future__ import print_function
from collections import defaultdict

import os
os.environ['KERAS_BACKEND']='theano'
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

K.set_image_dim_ordering('th')

def generator(latent_size):

	cnn = Sequential()

	cnn.add(Dense(1024, input_dim=latent_size, activation='relu'))
	cnn.add(Dense(128 * 7 * 7, activation='relu'))
	cnn.add(Reshape((128, 7, 7)))

	cnn.add(UpSampling2D(size=(2,2)))
	cnn.add(Conv2D(256, (5, 5), padding='same', activation='relu',kernel_initializer='glorot_normal'))

	cnn.add(UpSampling2D(size=(2,2)))
	cnn.add(Conv2D(128, (5, 5), padding='same', activation='relu',kernel_initializer='glorot_normal'))
	cnn.add(Conv2D(1, (2, 2), padding='same', activation='tanh',kernel_initializer='glorot_normal'))
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

	epochs = 5
	batch_size = 10
	latent_size = 10

	adam_lr = 0.0002
	adam_beta_1 = 0.5

	discriminator = discriminator()
	discriminator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
										loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])

	generator = generator(latent_size)
	generator.compile(optimizer=Adam(lr=adam_lr,beta_1=adam_beta_1),
										loss='binary_crossentropy')

	latent = Input(shape=(latent_size, ))
	image_class = Input(shape=(1,), dtype='int32')
	
	fake = generator([latent, image_class])

	discriminator.trainable = False
	fake, aux = discriminator(fake)

	combined = Model(outputs=[fake, aux], inputs=[latent, image_class])

	combined.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
					loss=['binary_crossentropy', 'sparse_categorical_crossentropy'])
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = (x_train.astype(np.float32) - 127.5)/ 127.5
	x_train = np.expand_dims(x_train, axis=1)

	x_test = (x_test.astype(np.float32) - 127.5) / 127.5
	x_test = np.expand_dims(x_test, axis=1)

	nb_train, nb_test = x_train.shape[0], x_test.shape[0]

	train_history = defaultdict(list)
	test_history = defaultdict(list)

	for epoch in range(epochs):
		print('Epoch {} of {}'.format(epoch + 1, epochs))

		batches = int(x_train.shape[0] / batch_size)
		progress_bar = Progbar(target=batches)

		epoch_gen_loss = []
		epoch_disc_loss = []

		for index in range(batches):

			progress_bar.update(index)

			noise = np.random.uniform(-1, 1, (batch_size, latent_size))

			image_batch = x_train[index * batch_size:(index + 1) * batch_size]
			label_batch = y_train[index * batch_size:(index + 1) * batch_size]

			sampled_labels = np.random.randint(0, 10, batch_size)

			generated_images = generator.predict([noise, sampled_labels.reshape((-1,1))], verbose=0)

			x = np.concatenate((image_batch, generated_images))
			y = np.array([1] * batch_size + [0] * batch_size)

			aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

			epoch_disc_loss.append(discriminator.train_on_batch(x, [y, aux_y]))

			noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
			sampled_labels = np.random.randint(0, 10, 2 * batch_size)

			trick = np.ones(2 * batch_size)

			epoch_gen_loss.append(combined.train_on_batch([noise, sampled_labels.reshape((-1,1))],
				[trick, sampled_labels]))

			print('\n Testing for epoch {}:'.format(epoch + 1))

			noise = np.random.uniform(-1, 1, (nb_test, latent_size))

			sampled_labels = np.random.randint(0 ,10, nb_test)
			generated_images = generator.predict([noise, sampled_labels.reshape((-1, 1))], verbose=False)

			x = np.concatenate((x_test, generated_images))
			y = np.array([1] * nb_test + [0] * nb_test)
			aux_y = np.concatenate((y_test, sampled_labels), axis=0)

			discriminator_test_loss = discriminator.evaluate(x, [y, aux_y], verbose=False)

			discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

			noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
			sampled_labels = np.random.randint(0, 10, 2 * nb_test)

			trick = np.ones(2 * nb_test)

			generator_test_loss = combined.evaluate([noise, sampled_labels.reshape((-1,1))], 
				[trick, sampled_labels], verbose=False)

			generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

			train_history['generator'].append(generator_train_loss)
			train_history['discriminator'].append(discriminator_train_loss)

			test_hisgory['generator'].append(generator_test_loss)
			test_history['discriminator'].append(discriminator_test_loss)

			generator.save_weights('params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
			discriminator.save_weights('params_generator_epoch_{0:03d}.hdf5'.format(epoch),True)

			noise = np.random.uniform(-1, 1, (100, latent_size))

			sampled_labels = np.array([ [i] * 10 for i in range(10)]).reshape(-1,1)

			generator_images = generator.predict([noise, sampled_labels],verbose=0)

			img = (np.concatenate([r.reshape(-1, 28) for r in np.split(generated_images, 10)], axis=-1)
				* 127.5 + 127.5).astype(np.uint8)

			Image.fromarray(img).save('plot_epoch_{0:03d}_generated.png'.format(epoch))

	pickle.dump({'train:' : train_history, 'test' : test_history}, open('acgan-history.pkl','wb'))




