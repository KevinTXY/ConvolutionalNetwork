import pickle
import tensorflow as tf

from tensorflow.keras import layers, models
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np


class AlexNet(object):
    def __init__(self, batch):
        self.X = batch

        first_conv = conv(batch, 96, 11, 4, 'conv_1', 'VALID', 0)
        norm_first_conv = resp_normalization(first_conv, 2, 5, 1e-4, .75, 'conv_1')
        first_max_pool = max_pooling(norm_first_conv, 3, 2, 'max_pool_1', 'VALID')

        second_conv = conv(first_max_pool, 256, 5, 1, 'conv_2', 'SAME', 1)
        norm_second_conv = resp_normalization(second_conv, 2, 5, 1e-4, .75, 'conv_2')
        second_max_pool = max_pooling(norm_second_conv, 3, 2, 'max_pool_2', 'VALID')
        third_conv = conv(second_max_pool, 384, 3, 1, 'conv_3', 'SAME', 0)

        fourth_conv = conv(third_conv, 384, 3, 1, 'conv_4', 'SAME', 1)
        
        fifth_conv = conv(fourth_conv, 256, 3, 1, 'conv_5', 'SAME', 1)
        third_max_pool = max_pooling(fifth_conv, 3, 2, 'max_pool_3', 'VALID')

        flat_size = tf.size(third_max_pool) / third_max_pool.get_shape()[0]
        flatten = tf.reshape(third_max_pool, [-1, int(flat_size)])
        first_fc = fully_connected(flatten, 4096, name = 'fully_connected_1')
        first_dropout = tf.nn.dropout(first_fc, 0.5)

        second_fc = fully_connected(first_dropout, 4096, name = 'fully_connected_2')
        second_dropout = tf.nn.dropout(second_fc, 0.5)

        # unnormalized, apply softmax later
        self.scores = fully_connected(second_dropout, 10, activation = False, name = 'fully_connected_3')


def conv(layer_input, filters, kernel_size, stride, name, padding = 'SAME', init_bias = 0):
    weight = tf.Variable(tf.random.normal(
        shape = [kernel_size, kernel_size, int(layer_input.get_shape()[-1]), filters],
        stddev=.01), name = name + '_kernels')
    bias = tf.Variable(init_bias * tf.ones(shape = [filters]), name = name + '_biases')
    conv_layer = tf.nn.conv2d(layer_input, weight, strides = [1, stride, stride, 1], padding = padding)
    conv_bias = tf.nn.bias_add(conv_layer, bias)
    return tf.nn.relu(conv_bias, name = name)


def resp_normalization(layer_input, k, depth_radius, alpha, beta, name):
    return tf.nn.local_response_normalization(layer_input, bias = k,
                                              depth_radius = depth_radius,
                                              alpha = alpha, beta = beta,
                                              name = name + '_norm')


def max_pooling(layer_input, pool_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(layer_input, ksize = [1, pool_size, pool_size, 1],
                          strides = [1, stride, stride, 1], padding = padding,
                          name = name)


def fully_connected(flattened_input, neurons, name, activation = True):
    weight = tf.Variable(tf.random.normal(
        shape = [int(flattened_input.get_shape()[-1]), neurons],
        stddev=.01), name = name + '_weights')
    bias = tf.Variable(tf.ones(shape = [neurons]), name = name + '_biases')
    out = tf.matmul(flattened_input, weight) + bias
    if activation:
        return tf.nn.relu(out, name = name)
    return out


def loadBatch(filename):
    path = 'datasets/cifar10/'
    with open(path + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        y = np.asarray(dict[b'labels'])
        X = np.asarray(dict[b'data']) / 255
        Y = to_categorical(y)   
    return X.T, Y.T, y

        
def main():
    Xtrain, Ytrain, ytrain = loadBatch('data_batch_1')
    Xtest, Ytest, ytest = loadBatch('test_batch')

    # should use sth like this for training
    # init_batch = tf.placeholder(shape=[128, 32, 32, 3])
    # X = tf.convert_to_tensor(Xtrain, np.float32)
    mod = AlexNet(tf.ones([23, 32, 32, 3]))
    print(mod.scores.get_shape())

if __name__ == '__main__':
    main()
