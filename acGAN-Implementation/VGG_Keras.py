from keras.models import Sequential
from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import h5py

def VGG_19(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,256,256)))
    print model.layers[-1].output_shape
    model.add(Convolution2D(64, (3,3), activation='relu'))
    print model.layers[-1].output_shape
    model.add(ZeroPadding2D((1,1)))
    print model.layers[-1].output_shape
    model.add(Convolution2D(8, (3,3), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(1,1)))
    print model.layers[-1].output_shape

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(1,1)))
    print model.layers[-1].output_shape
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3,3), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(1,1)))
    print model.layers[-1].output_shape

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(1,1)))
    print model.layers[-1].output_shape

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3,3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(2048, (3,3), activation='relu'))
    model.add(MaxPooling2D((1,1), strides=(1,1)))
    print model.layers[-1].output_shape


    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    f.close()

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    print model.layers[-1].output_shape

    return model