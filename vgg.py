# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io

"""
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)
"""
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1'
)


def load_net(data_path):
    """
    Loads weights from VGG. mean_pixel = ?
    """
    data = scipy.io.loadmat(data_path)
    if not all(i in data for i in ('layers', 'classes', 'normalization')):
        raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]

    return weights, mean_pixel
        


def net_preloaded(weights, input_image):
    """
    Generates layers of VGG net.
    """
    
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias, name)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net

def save_parameters(sess, extra_parameters):
    """
    Saves the parameters trained in sess, in file 'trained_vgg_net.m'
    extra_parameters is a dictionary of additional parameters to save
    """
    file_name = 'trained_vgg_net.m'
    
    graph = tf.get_default_graph()
    parameter_dict = {} 
    
    for i, layer_name in enumerate(VGG19_LAYERS):
        kind = layer_name[:4]
        if kind == 'conv':
            parameter_dict['w' + layer_name] = sess.run(graph.get_tensor_by_name('net/w' + layer_name + ':0'))
            parameter_dict['b' + layer_name] = sess.run(graph.get_tensor_by_name('net/b' + layer_name + ':0'))

    parameter_dict.update(extra_parameters)
            
    scipy.io.savemat(file_name, parameter_dict)

    return file_name

def _conv_layer(input, weights, bias, layer_name):

    w = tf.get_variable("w" + layer_name, initializer=tf.Variable(weights))
    b = tf.get_variable("b" + layer_name, initializer=tf.Variable(bias))
    
    conv = tf.nn.conv2d(input, w, strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, b)


def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel
