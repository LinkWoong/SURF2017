import cv2
from scipy import ndimage
import numpy as np 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import pandas as pd 
from keras.models import load_model
import tensorflow as tf 


def resize(image):

    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    return image
def load_image(path):

    images = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:

            images.append(img)
    return images

def light_map_single(image):

    image = image[None]
    image = image.transpose((1,2,0))
    blur = cv2.GaussianBlur(image, (0,0), 2)
    image = image.reshape((image.shape[0], image.shape[1]))

    result =(image.astype(int) - blur.astype(int)).astype(np.float) / 128.0

    return result
def normalize_pic(image):

    image = image.astype(np.float) / 255.0

    return image /np.max(image)

def resize_img(image):

    zeros = np.zeros((512, 512, image.shape[2]), dtype=np.float)
    image = zeros[:image.shape[0], :image.shape[1]]

    return image
def show_active_img_and_save(name,image,path):

    image = image.astype(np.float)
    image = image / 255.0
    image = -image + 1
    mat = image  * 255.0

    mat[mat<0] = 0 
    mat[mat>255] = 255

    mat = mat.astype(np.uint8)

    cv2.imwrite(path,mat)

    return image
def show_active_img_and_save_denoise(name,image,path):

    image = image.astype(np.float)
    image = image / 255.0
    image = -image + 1
    mat = image  * 255.0

    mat[mat<0] = 0 
    mat[mat>255] = 255

    mat = mat.astype(np.uint8)
    mat = ndimage.median_filter(mat,1)


    cv2.imwrite(path,mat)

    return image
def show_active_img_and_save_denoise_filter(name,image,path):
    mat = image.astype(np.float)
    mat[mat<0.18] = 0
    mat = -mat + 1
    mat = mat * 255.0
    mat[mat<0] = 0
    mat[mat>255] = 255
    mat = mat.astype(np.uint8)

    mat = ndimage.median_filter(mat,1)

    cv2.imwrite(path,mat)

    return mat


def show_active_img_and_save_denoise_filter2(name,image,path, i):

    path = path + '/sketch'
    mat = image.astype(np.float)
    mat[mat<0.1] = 0
    mat = -mat + 1
    mat = mat * 255.0
    mat[mat<0] = 0
    mat[mat>255] = 255
    mat = mat.astype(np.uint8)

    mat = ndimage.median_filter(mat,1)
    if not os.path.exists(path):
        os.makedirs(path)
    path = path + '/sketched_' + str(i) + '.jpeg'
    cv2.imwrite(path,mat)

    return mat

def resize_img_512_3d(img):
    zeros = np.zeros((1,3,512,512), dtype=np.float)
    zeros[0 , 0 : img.shape[0] , 0 : img.shape[1] , 0 : img.shape[2]] = img
    return zeros.transpose((1,2,3,0))
