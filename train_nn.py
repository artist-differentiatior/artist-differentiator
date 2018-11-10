import os
import time
from collections import OrderedDict

from PIL import Image
import numpy as np
import tensorflow as tf

import vgg

from load_images import *

STYLE_LAYERS = ('relu1_1', 'relu2_1')
#STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def train_nn(network, iterations, learning_rate, beta1, beta2, epsilon, batch_size=2):

    """
    Trains the neural net using triplet loss

    Args: 
        network: (?) pretrained neural network
        iterations: (int) number of training iterations
        learning_rate: (float) learning rate
        beta1: (float) momentum parameter for adam optimizer
        beta2: (float) RMSprop parameter for adam optimizer
        epsilon: (float) prevent division with 0 in adaom optimizer
        batch_size: (int) size of mini batches

    """

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    anchor_image = tf.placeholder('float', shape=(None, 224,224,3))
    positive_image = tf.placeholder('float', shape=(None, 224,224,3))
    negative_image = tf.placeholder('float', shape=(None, 224,224,3))
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        anchor_net = vgg.net_preloaded(vgg_weights, anchor_image)
        positive_net = vgg.net_preloaded(vgg_weights, positive_image)
        negative_net = vgg.net_preloaded(vgg_weights, negative_image)

    anchor_styles = _generate_style(anchor_net, STYLE_LAYERS)
    positive_styles = _generate_style(positive_net, STYLE_LAYERS)
    negative_styles = _generate_style(negative_net, STYLE_LAYERS)

    #loss_threshold = 1000000000

    dist_p = tf.add_n([tf.nn.l2_loss(anchor_styles[layer] - positive_styles[layer]) for layer in STYLE_LAYERS]) / batch_size
    dist_n = tf.add_n([tf.nn.l2_loss(anchor_styles[layer] - negative_styles[layer]) for layer in STYLE_LAYERS]) / batch_size
    loss = (dist_p - dist_n)

    # optimizer setup
    train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

    # Initialize image loader
    image_loader = Image_Loader('preprocessed_images/', batch_size)
    anchor, positive, negative = np.array(image_loader.load_next_batch())

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())


        #TODO: restore model
        """
        try:
            saver.restore(sess, "/home/albin/artist-differentiator/save/model.ckpt")
            print("Restored weights")
        except ValueError:
            print("Could not load .ckpt file")
            sess.run(tf.global_variables_initializer())
        """

        print('Optimization started...')
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
            
            #anchor, positive, negative = np.array(image_loader.load_next_batch())

            anchor = vgg.preprocess(anchor, vgg_mean_pixel)
            negative = vgg.preprocess(positive, vgg_mean_pixel)
            positive = vgg.preprocess(negative, vgg_mean_pixel)

            _, cost, cost2 = sess.run([train_step, dist_p, dist_n], feed_dict={anchor_image : anchor, positive_image: positive, negative_image: negative})
            print(cost, cost2)

            #TODO: save model

            """
            if (i + 1) % 5 == 0:
                save_path = saver.save(sess, "/home/albin/artist-differentiator/save/model.ckpt")
                print("Model saved in path: %s" % save_path)
            """
          
            iteration_end = time.time()
            iteration_times.append(iteration_end - iteration_start)

def _generate_style(net, style_layers):
    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[1] * features.shape[2], features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features, perm=[0,2,1]), features), tf.size(features[0], out_type=tf.float32))
        styles[layer] = gram

    return styles

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

