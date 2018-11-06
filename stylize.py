# Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3.

import os
import time
from collections import OrderedDict

from PIL import Image
import numpy as np
import tensorflow as tf

import vgg


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
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    style_features = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    # Builds the anchor graph
    # Anchor should be the first image in "styles"
    # TODO: test
    #anchor_graph = tf.Graph()
    #with anchor_graph.as_default(), anchor_graph.device('/cpu:0'):

    anchor_image = tf.placeholder('float', shape=style_shapes[0])
    positive_image = tf.placeholder('float', shape=style_shapes[1])
    negative_image = tf.placeholder('float', shape=style_shapes[2])
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        anchor_net = vgg.net_preloaded(vgg_weights, anchor_image, pooling)
        positive_net = vgg.net_preloaded(vgg_weights, positive_image, pooling)
        negative_net = vgg.net_preloaded(vgg_weights, negative_image, pooling)

    anchor_styles = _generate_style(anchor_net, STYLE_LAYERS)
    positive_styles = _generate_style(positive_net, STYLE_LAYERS)
    negative_styles = _generate_style(negative_net, STYLE_LAYERS)

    loss_threshold = 0.5

    # Loss = max(norm(A - P) - norm(A - N) + loss_threshold, 0)
    loss = tf.maximum(tf.add_n([tf.nn.l2_loss(anchor_styles[layer] - positive_styles[layer]) for layer in STYLE_LAYERS]) \
                      - tf.add_n([tf.nn.l2_loss(anchor_styles[layer] - negative_styles[layer]) for layer in STYLE_LAYERS]) + loss_threshold, 0)
        
    """
    # Not needed. Our loss is now one single quantity

    # We use OrderedDict to make sure we have the same order of loss types
    # (content, tv, style, total) as defined by the initial costruction of
    # the loss_store dict. This is important for print_progress() and
    # saving loss_arrs (column order) in the main script.
    #
    # Subtle Gotcha (tested with Python 3.5): The syntax
    # OrderedDict(key1=val1, key2=val2, ...) does /not/ create the same
    # order since, apparently, it first creates a normal dict with random
    # order (< Python 3.7) and then wraps that in an OrderedDict. We have
    # to pass in a data structure which is already ordered. I'd call this a
    # bug, since both constructor syntax variants result in different
    # objects. In 3.6, the order is preserved in dict() in CPython, in 3.7
    # they finally made it part of the language spec. Thank you!
    loss_store = OrderedDict([('content', content_loss),
                              ('style', style_loss),
                              ('tv', tv_loss),
                              ('total', loss)])
    """

    

    # optimizer setup
    # TODO: try adding triplet images via feed_dict
    train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

    # TODO: Enter batches of images in the loop
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

            style_pre = np.array([vgg.preprocess(styles[0], vgg_mean_pixel)])
            
            train_step.run(feed_dict={anchor_image : style_pre, positive_image: style_pre, negative_image: style_pre})

            last_step = (i == iterations - 1)
            if last_step or (print_iterations and i % print_iterations == 0):
                loss_vals = get_loss_vals(loss_store)
                print_progress(loss_vals)
            else:
                loss_vals = None

            if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                this_loss = loss.eval()
                if this_loss < best_loss:
                    best_loss = this_loss
                    best = image.eval()

                img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                if preserve_colors and preserve_colors == True:
                    original_image = np.clip(content, 0, 255)
                    styled_image = np.clip(img_out, 0, 255)

                    # Luminosity transfer steps:
                    # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                    # 2. Convert stylized grayscale into YUV (YCbCr)
                    # 3. Convert original image into YUV (YCbCr)
                    # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                    # 5. Convert recombined image from YUV back to RGB

                    # 1
                    styled_grayscale = rgb2gray(styled_image)
                    styled_grayscale_rgb = gray2rgb(styled_grayscale)

                    # 2
                    styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

                    # 3
                    original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

                    # 4
                    w, h, _ = original_image.shape
                    combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                    combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                    combined_yuv[..., 1] = original_yuv[..., 1]
                    combined_yuv[..., 2] = original_yuv[..., 2]

                    # 5
                    img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))
            else:
                img_out = None

            yield i+1 if last_step else i, img_out, loss_vals

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
