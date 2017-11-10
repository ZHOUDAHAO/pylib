import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import time
import no_warnings
from datetime import datetime as dt
from auxi import print_separator

def get_summaries_dir():
	time_part = dt.now().strftime("%d %H:%M:%S").split()
	summaries_dir = '_'.join([time_part[0], time_part[1].replace(':','')])
	return summaries_dir

def get_ele_num(shape):
	num = 1
	for s in shape:
		num *= s
	return num

# W = [filter_width, in_channels, out_channels]
def conv1d(x, W, stride, padding = 'SAME', Name = None):
	return tf.nn.conv1d(x, W, stride = stride, padding = padding, name = Name)

def conv2d(x, W, stride, padding = 'SAME', Name = None):
	return tf.nn.conv2d(x, W, strides = [1,stride,stride,1], padding = padding, name = Name)

def conv2d_as_1d(x, W, stride, padding = 'SAME', Name = None):
	return tf.nn.conv2d(x, W, strides = [1,stride,1,1], padding = padding, name = Name)

def	max_pool2d(x, ks, stride, padding = 'SAME', Name = None):
	return tf.nn.max_pool(x, ksize = [1,ks,ks,1], strides = [1,stride,stride,1], padding = padding, name = Name)

def max_pool1d(x, ks, stride, Name = None):
	_x = tf.expand_dims(x, [2])
	_output = tf.nn.max_pool(_x, ksize = [1, ks, 1, 1], strides = [1, stride, 1, 1], padding = 'SAME', name = Name)
	return tf.squeeze(_output, [2])

def max_pool2d_as_1d(x, ks, stride, Name = None):
	return tf.nn.max_pool(x, ksize = [1, ks, 1, 1], strides = [1, stride, 1, 1], padding = 'SAME', name = Name)

def conv_bia_nonlin(inputs, conv, stride, bias = None, convfunc = conv2d, name1 = None, name2 = None):
	_conv = convfunc(inputs, conv, stride, Name = name1)
	if bias is not None:
		_conv = tf.nn.bias_add(_conv, bias, name = name2)
	output = tf.nn.relu(_conv)
	return _conv, output

def conv_bn(inputs, conv, stride, convfunc, relu):
	if relu not in ['First','Second','None']:
		raise Exception('relu argument should be "None","First" or "Second"')
	if relu == 'First':
		inputs = tf.nn.relu(inputs)
	_conv = convfunc(inputs, conv, stride)
	output = keep_dim_batchnorm(_conv)
	if relu == 'Second':
		output = tf.nn.relu(output)
	return output

def set_random_seed():
	seed = int(time.time()) % 1000000
	tf.set_random_seed(seed)

def myinit(name, shape):
	np.random.seed(int(time.time()))
	init = tf.constant(0.01 * np.random.randn(*shape))
	init = tf.cast(init, tf.float32)
	res = tf.get_variable(name, initializer = init)
	return res

def xaiver_init(name, shape):
	return tf.get_variable(name, shape = shape, dtype = tf.float32, initializer = xavier_initializer())

def gaussian_variable(name, shape):
	set_random_seed()
	mean = np.random.uniform(-2.0, 2.0, (1,))
	stddev = np.random.uniform(2.0, 4.0, (1,))
	initial = tf.truncated_normal(shape, name = name, mean = mean, stddev = stddev)
	return tf.Variable(initial)

def constant_variable(shape):
	set_random_seed()
	bias = tf.constant(0.1, shape = shape)
	return tf.Variable(bias)

# start and end are the start and end row number
def print_tensor(variable, session, feed_dict, name = None, start = None, end = None):
	if name is not None:
		print name
	else:
		print 'name',variable.name
	if len(variable.get_shape()) <= 1:
		print session.run(variable, feed_dict = feed_dict)
	else:
		if (not start) and (not end):
			v = variable
		elif not start:
			v = variable[:end]
		elif not end:
			v = variable[start:]
		else:
			v = variable[start:end]
		print session.run(v, feed_dict = feed_dict)
	print 

# matrix and scalar allclose, atol = absolute tolerance
def ma_sc_allclose(a, val, atol = 1e-6):
	def f(x):
		return 1.0 if abs(x - val) < atol else 0.0
	f = np.vectorize(f, otypes=[np.float])
	return f(a)

def count_zero_rate(a):
	return np.sum(ma_sc_allclose(a, 0.0)) / a.size

def print_learn_args(Config):
	print 'Train times',Config.train_times
	print 'Mini batch size',Config.mini_bsize
	print 'Try learning rate',Config.lr
	print_separator()

# one_dim is the order of dimension whose dimension is 1
def keep_dim_batchnorm(x, one_dim = 2):
	if x.get_shape()[one_dim] == tf.Dimension(1):
		x = tf.squeeze(x, one_dim)
		x = tf.contrib.layers.batch_norm(x)
		x = tf.expand_dims(x, one_dim)
	else:
		x = tf.contrib.layers.batch_norm(x)
	return x

def crop(a, b, offset):
	hb = int(b.get_shape()[-3])
	wb = int(b.get_shape()[-2])
	ha = int(a.get_shape()[-3])
	wa = int(a.get_shape()[-2])
	y1 = offset / float(ha - 1)
	x1 = offset / float(wa - 1)
	y2 = (offset + hb - 1) / float(ha - 1)
	x2 = (offset + wb - 1) / float(wa - 1)
	print ha,wa,hb,wb
	print x1,y1,x2,y2
	print hb,wb
	res = tf.image.crop_and_resize(a, [[y1, x1, y2, x2]], [0], [hb, wb]) 
	return res

def deconv2d(x, W, stride, padding = 'SAME'):
	h = x.get_shape()[-3]
	w = x.get_shape()[-2]
	return tf.nn.conv2d_transpose(x, W, [1, stride * h, stride * w, 1], [1, stride, stride, 1], padding = padding)
