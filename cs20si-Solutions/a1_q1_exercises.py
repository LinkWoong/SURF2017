import os 
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np 
###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.less(x, y), lambda: tf.add(x, y), lambda: tf.sub(x, y))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

# YOUR CODE
x = tf.random_uniform(minval=-1,maxval=1)
y = tf.random_uniform(minval=-1,maxval=1)
out = tf.case({tf.less(x,y): x+y, tf.greater(x,y), x-y},default=0)

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################
# YOUR CODE
x = tf.get_variable('x',[[0,-2,-1],[0,1,2]],initializer=tf.truncated_normal_initializer())
y = tf.get_variable('y',np.zeros(tf.size(x)),initializer=tf.truncated_normal_initializer())
out = tf.equal(x,y)

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################
x = tf.Variable([29.05088806,  27.61298943,  31.19073486,  29.35532951,
 30.97266006,  26.67541885,  38.08450317,  20.74983215,
 34.94445419,  34.45999146,  29.06485367,  36.01657104,
 27.88236427,  20.56035233,  30.20379066,  29.51215172,
 33.71149445,  28.59134293,  36.05556488,  28.66994858],dtype=tf.float32,trainable=True,name='x')
con = lambda x: x>30
y = tf.where(con,x=x,y=None,name='indices')
out = tf.gather(params=x,indices=y,validate_indices=True,name='gather')

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

# YOUR CODE
diagnoal = [1,2,3,4,5,6]
out = tf.Variable(tf.diag(diagnoal),dtype=tf.float32, trainable=True,name='output')


###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################
x = tf.Variable(tf.random_normal(shape=[10,10],mean=0.0, stddev=12.0,dtype=tf.float32,seed=None,name='random'),dtype=tf.float32,trainable=True,name='x')
out = tf.matrix_determinant(x,name='output')

# YOUR CODE

###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

x = tf.Variable([5,2,3,5,10,6,2,3,4,2,1,1,0,9],dtype=tf.float32,trainable=True,name='x')
out = tf.unique(x,name='output',control_inputs=None)

# YOUR CODE

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################
x = tf.Variable(tf.random_normal(shape=300,mean=0.0, stddev=12.0, dtype=tf.float32),dtype=tf.float32,name='x')
y = tf.Variable(tf.random_normal(shape=300,mean=0.0, stddev=12.0, dtype=tf.float32),dtype=tf.float32,name='y')
diff = tf.add_n(x-y)/tf.size(x-y)
f1 = lambda: tf.losses.mean_squared_error(x,y,weights=1.0)
f2 = lambda: tf.add_n(tf.abs(diff))
out = tf.case([(tf.less(diff,0),f1)],default=f2)