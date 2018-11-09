# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.

import os
import time
from collections import OrderedDict

from PIL import Image
import numpy as np
import tensorflow as tf

import vgg

from load_images import *


STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


try:
    reduce
except NameError:
    from functools import reduce


def get_loss_vals(loss_store):
    return OrderedDict((key, val.eval()) for key,val in loss_store.items())


def print_progress(loss_vals):
    for key,val in loss_vals.items():
        print('{:>13s} {:g}'.format(key + ' loss:', val))


def stylize(network, initial, initial_noiseblend, content, styles, preserve_colors, iterations,
        content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
        learning_rate, beta1, beta2, epsilon, pooling,
        print_iterations=None, checkpoint_iterations=None):
    """
    Stylize images.

    This function yields tuples (iteration, image, loss_vals) at every
    iteration. However `image` and `loss_vals` are None by default. Each
    `checkpoint_iterations`, `image` is not None. Each `print_iterations`,
    `loss_vals` is not None.

    `loss_vals` is a dict with loss values for the current iteration, e.g.
    ``{'content': 1.23, 'style': 4.56, 'tv': 7.89, 'total': 13.68}``.

    :rtype: iterator[tuple[int,image]]
    """
    
    minibatch_size = 16
    
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    # Builds the anchor graph
    # Anchor should be the first image in "styles"
    # TODO: test
    #anchor_graph = tf.Graph()
    #with anchor_graph.as_default(), anchor_graph.device('/cpu:0'):

    anchor_image = tf.placeholder('float', shape=(minibatch_size, 224,224,3))
    positive_image = tf.placeholder('float', shape=(minibatch_size, 224,224,3))
    negative_image = tf.placeholder('float', shape=(minibatch_size, 224,224,3))
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        anchor_net = vgg.net_preloaded(vgg_weights, anchor_image, pooling)
        positive_net = vgg.net_preloaded(vgg_weights, positive_image, pooling)
        negative_net = vgg.net_preloaded(vgg_weights, negative_image, pooling)

    anchor_styles = _generate_style(anchor_net, STYLE_LAYERS)
    positive_styles = _generate_style(positive_net, STYLE_LAYERS)
    negative_styles = _generate_style(negative_net, STYLE_LAYERS)

    loss_threshold = 100

    # Loss = max(norm(A - P) - norm(A - N) + loss_threshold, 0)
    loss = tf.maximum(tf.add_n([tf.nn.l2_loss(anchor_styles[layer] - positive_styles[layer]) for layer in STYLE_LAYERS]) \
                      - tf.add_n([tf.nn.l2_loss(anchor_styles[layer] - negative_styles[layer]) for layer in STYLE_LAYERS]) + loss_threshold, 0)

    
    # optimizer setup
    train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

    # Initialize image loader
    image_loader = Image_Loader('preprocessed_images/', minibatch_size)
    
    # optimization
    best_loss = float('inf')
    best = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('Optimization started...')
        if (print_iterations and print_iterations != 0):
            print_progress(get_loss_vals(loss_store))
        iteration_times = []
        start = time.time()
        for i in range(iterations):
            iteration_start = time.time()
            if i > 0:
                elapsed = time.time() - start
                # take average of last couple steps to get time per iteration
                remaining = np.mean(iteration_times[-10:]) * (iterations - i)
                print('Iteration %4d/%4d (%s elapsed, %s remaining)' % (
                    i + 1,
                    iterations,
                    hms(elapsed),
                    hms(remaining)
                ))
            else:
                print('Iteration %4d/%4d' % (i + 1, iterations))

            style_pre0 = np.array([vgg.preprocess(styles[0], vgg_mean_pixel)])
            style_pre1 = np.array([vgg.preprocess(styles[1], vgg_mean_pixel)])
            style_pre2 = np.array([vgg.preprocess(styles[2], vgg_mean_pixel)])

            anchor, positive, negative = np.array(image_loader.load_next_batch())

            anchor = np.array([vgg.preprocess(anchor, vgg_mean_pixel)])
            positive = np.array([vgg.preprocess(positive, vgg_mean_pixel)])
            negative = np.array([vgg.preprocess(negative, vgg_mean_pixel)])
            
            train_step.run(feed_dict={anchor_image : anchor, positive_image: positive, negative_image: negative})
            print(loss.eval(feed_dict={anchor_image : anchor, positive_image: positive, negative_image: negative}))


            """
            last_step = (i == iterations - 1)
            if last_step or (print_iterations and i % print_iterations == 0):
                loss_vals = get_loss_vals(loss_store)
                print_progress(loss_vals)
            else:
                loss_vals = None

            """


          
            iteration_end = time.time()
            iteration_times.append(iteration_end - iteration_start)

def _generate_style(net, style_layers):
    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features), features), tf.size(features, out_type=tf.float32))
        styles[layer] = gram

    return styles

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return '%d hr %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds

