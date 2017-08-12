import tensorflow as tf
import numpy as np
import io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import cv2
import pandas as pd 


from keras.models import load_model
from preprocess import *

path = '/media/linkwong/D/1girl'
mod_path = '/media/linkwong/D/mod.h5'

content_to_sketch = True


content_path, sketch_path = preprocess(path, mod_path, content_to_sketch)

print content_path
print sketch_path
