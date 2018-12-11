import scipy.io

import tensorflow as tf
import numpy as np

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
    Loads parameters from data_path (path to .mat file)

    Args:
        data_path: (string) path to saved net
    """

    parameter_dict = scipy.io.loadmat(data_path)
    del parameter_dict['__version__']
    del parameter_dict['__globals__']
    del parameter_dict['__header__']
    
    return parameter_dict


def net_preloaded(parameter_dict, input_image, num_freeze=0):

    """
    Generates layers of VGG net. Returns a dictionary of the neural net

    Args:
        parameter_dict: (dict) Dictionary holding all the weights for the network
        input_image: (tf tensor) Placegolder tensor with shape of input image
        num_freeze: (int) number of layers to freeze in network
    """
    
    net = {}
    current = input_image

    for i, layer_name in enumerate(VGG19_LAYERS):

        if i < num_freeze:
            freeze = True
        else:
            freeze = False

        kind = layer_name[:4]

        if kind == 'conv':

            weights = parameter_dict['w' + layer_name]
            bias = parameter_dict['b' + layer_name].reshape(-1)
            current = _conv_layer(current, weights, bias, layer_name, freeze)
        
        elif kind == 'relu':
            current = tf.nn.relu(current)

        elif kind == 'pool':
            current = _pool_layer(current)

        net[layer_name] = current

    assert len(net) == len(VGG19_LAYERS)

    return net


def _conv_layer(input, weights, bias, layer_name, freeze):

    """
    Create a conv-layer from the input parameters and returns it.

    Args:
        input: (tensor) Input to the conv layer
        weights: (numpy array) Weight matrix for conv layer
        bias: (numpy array) Bais vector for layer
        layer_name: (string) Name of the layer
        freeze: (bool) tells whether layer should be constant or trainable
    """

    if freeze:
        conv = tf.nn.conv2d(input, tf.constant(weights, name="w" + layer_name), strides=(1, 1, 1, 1),
            padding='SAME')
        bias = tf.constant(bias, name="b" + layer_name)
        return tf.nn.bias_add(conv, bias)

    else:
        w = tf.get_variable("w" + layer_name, initializer=tf.Variable(weights))
        b = tf.get_variable("b" + layer_name, initializer=tf.Variable(bias))
    
        conv = tf.nn.conv2d(input, w, strides=(1, 1, 1, 1),
                padding='SAME')
    
        return tf.nn.bias_add(conv, b)


def _pool_layer(input):

    """
    Create a and returns a max pool layer

    Args:
        input: (tensor) Input to the max pool layer
    """

    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')


def preprocess(image, mean_pixel):

    """
    Subtract mean pixel value to an image

    Args:
        image: (numpy array) Image to subtract mean from
        mean_pixel: (int) mean pixel value to subtract
    """

    return image - mean_pixel


def unprocess(image, mean_pixel):

    """
    Subtract mean pixel value to an image

    Args:
        image: (numpy array) Image to subtract mean from
        mean_pixel: (int) mean pixel value to subtract
    """

    return image + mean_pixel


def save_parameters(sess, extra_parameters, file_name):

    """
    Saves the parameters trained in sess, in file 'trained_vgg_net.m'
    extra_parameters is a dictionary of additional parameters to save

    Args:
        sess: (tf session) Session to save computational graph from
        extra_parameters: (dict) Contains extra parameters neccessary for the network
        file_name: (string) Name of file
    """
    
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
