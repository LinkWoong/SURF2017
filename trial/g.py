import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
from keras.applications import VGG19
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import losses
from keras.backend import tf as ktf
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers.merge import add
import matplotlib.pyplot as plt


def main_model():
    sketch = Input((512, 512, 1))
    style = Input((512, 512, 3))
    # The convolutional layers in the front
    conv1 = Conv2D(filters=16, kernel_size=2, strides=2, padding='SAME', activation='relu')(sketch)
    conv2 = Conv2D(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv2)
    conv4 = Conv2D(filters=128, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv4)
    conv6 = Conv2D(filters=2048, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv5)

    vgg_style = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(style)
    vgg_layer = VGG19(input_tensor=vgg_style, input_shape=(224, 224, 3)).output
    print vgg_layer.shape

    deconv7 = add([vgg_layer, conv6])
    deconv7 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    deconv7 = concatenate([deconv7, conv5], axis=3)
    deconv8 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    deconv8 = concatenate([deconv8, conv4], axis=3)
    deconv9 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv8)
    deconv9 = concatenate([deconv9, conv3], axis=3)
    deconv10 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv9)
    deconv10 = concatenate([deconv10, conv2], axis=3)
    deconv11 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv10)
    deconv11 = concatenate([deconv11, conv1], axis=3)

    output_layer = Dense(units=3)(deconv11)
    # style_256 = Lambda(lambda image: ktf.image.resize_images(image, (256, 256)))(style)
    # (style_256, output_layer)
    model = Model(inputs=[sketch, style], outputs=output_layer)
    model.compile(optimizer='sgd', loss=losses.mean_absolute_error, metrics=['accuracy'])

    # front_deco_model = front_decoder()
    # gray_style = ktf.image.rgb_to_grayscale(style)
    # front_deco_model.compile(optimizer=sgd, loss=losses.mean_absolute_error(gray_style, ), metrics=['accuracy'])
    # end_deco_model = end_decoder()
    # end_deco_model.compile(optimizer=sgd, loss=losses.mean_absolute_error(), metrics=['accuracy'])
    return model


def end_decoder():
    sketch = Input((512, 512, 1))
    style = Input((512, 512, 3))
    conv1 = Conv2D(filters=16, kernel_size=2, strides=2, padding='SAME', activation='relu')(sketch)
    print(conv1.shape)
    conv2 = Conv2D(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv2)
    conv4 = Conv2D(filters=128, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv4)
    conv6 = Conv2D(filters=2048, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv5)
    vgg_layer = VGG19(input_tensor=style, input_shape=(224, 224, 3)).output
    deconv7 = add([vgg_layer, conv6])
    deconv7 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    guide6 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="SAME", activation='relu')(deconv7)
    guide7 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide6)
    guide8 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide7)
    guide9 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide8)
    guide10 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide9)
    output_layer = Dense(3)(guide10)

    model = Model(inputs=[sketch, style], output=output_layer)
    model.compile(loss=losses.mean_absolute_error(output_layer, style), optimizer='sgd', metrics=['accuracy'])
    return model


def front_decoder():
    sketch = Input((512, 512, 1))
    style = Input((512, 512, 3))
    conv1 = Conv2D(filters=16, kernel_size=2, strides=2, padding='SAME', activation='relu')(sketch)
    conv2 = Conv2D(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv2)
    conv4 = Conv2D(filters=128, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv4)
    guide1 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="SAME", activation="relu")(conv5)
    guide2 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide1)
    guide3 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide2)
    guide4 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide3)
    guide5 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide4)
    output_layer = Dense(1)(guide5)

    gray_style = ktf.image.rgb_to_grayscale(style)
    model = Model(inputs=[sketch, style], output=output_layer)
    model.compile(loss=losses.mean_absolute_error(gray_style, output_layer), optimizer='sgd', metrics=['accuracy'])
    return model


def generator():
    sketch = Input((512, 512, 1))
    style = Input((512, 512, 3))

    network_model = main_model()
    network_loss = network_model.fit((sketch, style), batch_size=1).history.get('loss')

    front_deco_model = front_decoder()
    front_deco_loss = front_deco_model.fit((sketch, style), batch_size=1).history.get('loss')

    end_deco_model = end_decoder()
    end_deco_loss = end_deco_model.fit((sketch, style), batch_size=1).history.get('loss')

    # model = Model(inputs=[sketch, style], output=fake_img)
    # return model


if __name__ == '__main__':
    sketch_image = img_to_array(load_img('./Sketch.jpg', grayscale=True))
    sketch_image = K.expand_dims(sketch_image, axis=0)
    print(sketch_image.shape)
    style_image = img_to_array(load_img('./Style.jpg'))
    style_image = K.expand_dims(style_image, axis=0)
    print(style_image.shape)
    network = main_model()
    network.fit(sketch_image,style_image, batch_size=1)
