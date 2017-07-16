from keras_vgg19 import VGG19
import os   
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import matplotlib.pyplot as plt
import pylab
from keras import losses
from keras.backend import tf as ktf
from keras.layers import *
from keras.layers.merge import add
from keras.models import *
from keras.preprocessing.image import img_to_array, load_img

from keras_vgg19 import VGG19


def main_model():
    sketch = Input((512, 512, 1), name="sketch_input")
    style = Input((512, 512, 3), name="style_input")
    # The convolutional layers in the front
    conv1 = Conv2D(filters=16, kernel_size=2, strides=2, padding='SAME', activation='relu')(sketch)
    conv2 = Conv2D(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv2)
    conv4 = Conv2D(filters=128, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv3)
    conv5 = Conv2D(filters=256, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv4)
    conv6 = Conv2D(filters=2048, kernel_size=2, strides=2, padding="SAME", activation='relu')(conv5)
    # The front decoder
    guide1 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="SAME", activation="relu")(conv5)
    guide2 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide1)
    guide3 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide2)
    guide4 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide3)
    guide5 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide4)
    front_decoder_output = Dense(1, name="front_decoder_output")(guide5)
    # The vgg model
    vgg_style = Lambda(lambda image: ktf.image.resize_images(image, (224, 224)))(style)
    vgg_layer = VGG19(input_tensor=vgg_style, input_shape=(224, 224, 3)).output
    # The latter part of network
    deconv7 = add([vgg_layer, conv6])
    deconv7 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    # The end decoder
    guide6 = Conv2DTranspose(filters=512, kernel_size=2, strides=2, padding="SAME", activation='relu')(deconv7)
    guide7 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide6)
    guide8 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide7)
    guide9 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide8)
    guide10 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding="SAME", activation="relu")(guide9)
    end_decoder_output = Dense(3, name="end_decoder_output")(guide10)
    # The end part of the network
    deconv7 = concatenate([deconv7, conv5], axis=3)
    deconv8 = Conv2DTranspose(filters=256, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv7)
    deconv8 = concatenate([deconv8, conv4], axis=3)
    deconv9 = Conv2DTranspose(filters=128, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv8)
    deconv9 = concatenate([deconv9, conv3], axis=3)
    deconv10 = Conv2DTranspose(filters=64, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv9)
    deconv10 = concatenate([deconv10, conv2], axis=3)
    deconv11 = Conv2DTranspose(filters=32, kernel_size=2, strides=2, padding='SAME', activation='relu')(deconv10)
    deconv11 = concatenate([deconv11, conv1], axis=3)
    network_output = Dense(3, name="network_output")(deconv11)
    # The two form of style picture (used to calculate the loss)
    style_256 = Lambda(lambda image: ktf.image.resize_images(image, (256, 256)))(style)
    gray_style = ktf.image.rgb_to_grayscale(style)
    # The model part
    model = Model(inputs=[sketch, style], outputs=[front_decoder_output, end_decoder_output, network_output])
    # the ratio [1, 0.3, 0.9] is suggested in the paper
    model.compile(optimizer='sgd', loss={
        'front_decoder_output': losses.mean_absolute_error,
        'end_decoder_output': losses.mean_absolute_error,
        'network_output': losses.mean_absolute_error
    }, loss_weights=[1, 0.3, 0.9])
    # The true loss --- waited to be implemented
    # 'dense_1': losses.mean_absolute_error(gray_style, front_decoder_output),
    # 'dense_3': losses.mean_absolute_error(style, end_decoder_output),
    # 'dense_4': losses.mean_absolute_error(style_256, network_output)
    return model

if __name__ == '__main__':
    sketch_image = img_to_array(load_img('./Sketch.jpg', grayscale=True))
    sketch_image = np.expand_dims(sketch_image, axis=0)
    style_image = img_to_array(load_img('./Style.jpg'))
    style_image = np.expand_dims(style_image, axis=0)
    network = main_model()
    generate_image_list = network.predict({'sketch_input': sketch_image, 'style_input': style_image})
    [front_image, end_image, network_image] = generate_image_list

    front_image = front_image.reshape([512, 512])
    plt.imshow(front_image, cmap='gray')
    pylab.show()

    network_image = network_image.reshape([256, 256, 3])
    plt.imshow(network_image)
    pylab.show()

    end_image = end_image.reshape([512, 512, 3])
    plt.imshow(end_image)
pylab.show()