import tensorflow as tf
import os   
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# input has shape of 256*256*1
# the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32,[None,512,512,1])
# output has shape of 16*16*256
Y = tf.placeholder(tf.float32, [None,16,16,256])
# learning_rate
lr = tf.placeholder(tf.float32)

K = 16  # first convolutional layer output depth
L = 32 # second convolutional layer output depth
M = 64  # third convolutional layer
N = 128  # fourth convolutional layer
O = 256  # fifth convolutional layer
stride = 2 
# 2x2 patch, 1 input channel, K output channels
W1 = tf.Variable(tf.truncated_normal([2, 2, 1, K], stddev=0.1))
B1 = tf.Variable(tf.ones([K])/10)
W2 = tf.Variable(tf.truncated_normal([2, 2, K, L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)
W3 = tf.Variable(tf.truncated_normal([2, 2, L, M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)
W4 = tf.Variable(tf.truncated_normal([2, 2, M, N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)
W5 = tf.Variable(tf.truncated_normal([2, 2, N, O], stddev=0.1))
B5 = tf.Variable(tf.ones([O])/10)

Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1) # 256*256*1 -> 256*256*16
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2) # 256*256*16 -> 128*128*32
Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3) # 128*128*32 -> 64*64*64
Y4 = tf.nn.relu(tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='SAME') + B4) # 64*64*64 -> 32*32*128
Y5 = tf.nn.relu(tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='SAME') + B5) # 32*32*128 -> 16*16*256
Output = tf.reshape(Y5, shape=[8,8,2048]) #16*16*256 -> 8*8*2048


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)