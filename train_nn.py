import os
import time
import datetime
from collections import OrderedDict
from tqdm import tqdm # progressbar

from PIL import Image
import numpy as np
import tensorflow as tf

import trained_vgg

from load_images import *
import logging


NAME_STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
PREPROCESSED_PATH = './preprocessed_images/'

def train_nn(network, epochs, learning_rate, beta1, beta2, epsilon, save_file_name, checkpoints, loss_threshold,\
             positive_weight, batch_size, device_name, style_layers_indices):

    """
    Trains the neural net using triplet loss

    Args: 
        network: (str) filepath to pretrained network parameters
        epochs: (int) number of training epochs
        learning_rate: (float) learning rate
        beta1: (float) momentum parameter for adam optimizer
        beta2: (float) RMSprop parameter for adam optimizer
        epsilon: (float) prevent division with 0 in adaom optimizer
        save_file_name: (str) name of trained weights
        checkpoints: (bool) True if resume from checkpoints
        loss_threshold: (int) Extra threshold in loss function
        positive_weight: (float) weight for positive distance in loss function
        batch_size: (int) size of mini batches
        device_name: (str) which device to run computation graph on
        style_layers_indices: (array) which layers to extract style from

    """

    style_layers = [NAME_STYLE_LAYERS[i] for i in style_layers_indices]

    parameter_dict = trained_vgg.load_net(network)
    vgg_mean_pixel = parameter_dict['mean_pixel']

    anchor_image = tf.placeholder('float', shape=(None, 224,224,3))
    positive_image = tf.placeholder('float', shape=(None, 224,224,3))
    negative_image = tf.placeholder('float', shape=(None, 224,224,3))
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE), tf.device(device_name):
        anchor_net = trained_vgg.net_preloaded(parameter_dict, anchor_image, 19)
        positive_net = trained_vgg.net_preloaded(parameter_dict, positive_image, 19)
        negative_net = trained_vgg.net_preloaded(parameter_dict, negative_image, 19)

    anchor_styles = _generate_style(anchor_net, style_layers)
    positive_styles = _generate_style(positive_net, style_layers)
    negative_styles = _generate_style(negative_net, style_layers)

    dist_p = tf.add_n([tf.reduce_sum((anchor_styles[layer] - positive_styles[layer]) ** 2,[1,2]) for layer in style_layers])
    dist_n = tf.add_n([tf.reduce_sum((anchor_styles[layer] - negative_styles[layer]) ** 2,[1,2]) for layer in style_layers])
    
    max_sum = tf.maximum(positive_weight*dist_p - dist_n + loss_threshold, 0)
    loss = tf.reduce_sum(max_sum) / batch_size # Divide by batch size


    # Used to compute threshold for evaluating on pairs of images
    batch_avg_dist_AP = tf.reduce_sum(dist_p) / batch_size
    batch_avg_dist_AN = tf.reduce_sum(dist_n) / batch_size

    # optimizer setup
    train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

    # Initialize image loader
    image_loader = Image_Loader(PREPROCESSED_PATH, batch_size)

    saver = tf.train.Saver() 

    if not os.path.exists("Log"):
        os.makedirs("Log")
        print("Created directory Log")

    datetime_log = str(datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))

    logging.basicConfig(filename='Log/' + datetime_log + '.log',level=logging.INFO)
    logging.info('Started training: ' + datetime_log)

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        print("Created directory checkpoints")


    with tf.Session() as sess:
        
        if checkpoints:
            
            saver.restore(sess, "checkpoints/model.ckpt")

        else:   

            sess.run(tf.global_variables_initializer())

        graph = tf.get_default_graph()
        

        print('Optimization started...')
        epoch_times = []
        start = time.time()

        checkpoint_counter = 0

        for i in range(epochs):
            epoch_start = time.time()
            if i > 0:
                elapsed = time.time() - start
                # take average of last couple steps to get time per iteration
                remaining = np.mean(epoch_times[-10:]) * (epochs - i)
                epoch_info = 'Epoch %4d/%4d (%s elapsed, %s remaining)' % (
                    i + 1,
                    epochs,
                    hms(elapsed),
                    hms(remaining)
                )
                print(epoch_info)
                logging.info(epoch_info)
            else:
                print('Epoch %4d/%4d' % (i + 1, epochs))
            
            iteration_num = 0
            for anchor, positive, negative in tqdm(image_loader):
                
                anchor = trained_vgg.preprocess(anchor, vgg_mean_pixel)
                positive = trained_vgg.preprocess(positive, vgg_mean_pixel)
                negative = trained_vgg.preprocess(negative, vgg_mean_pixel)

                cost, _ = sess.run([loss, train_step], feed_dict={anchor_image : anchor, positive_image: positive, negative_image: negative})

                iteration_num += 1
                logging.info("Epoch: " + str(i + 1) + "/" + str(epochs) + \
                             " Iteration: " + str(iteration_num) + "/" + str(len(image_loader)) + \
                             " Cost: " + str(cost))

                if checkpoint_counter == 4:
                    save_path = saver.save(sess, "checkpoints/model.ckpt")
                    #print("Model saved in path: %s" % save_path)
                    checkpoint_counter = 0
                else:
                    checkpoint_counter += 1

            print('Cost: %e' % cost)
          
            epoch_end = time.time()
            epoch_times.append(epoch_end - epoch_start)

        print('Training completed. Computing mean distances...')

        avg_dist_AP = 0
        avg_dist_AN = 0
        for anchor, positive, negative in tqdm(image_loader):

            anchor = trained_vgg.preprocess(anchor, vgg_mean_pixel)
            positive = trained_vgg.preprocess(positive, vgg_mean_pixel)
            negative = trained_vgg.preprocess(negative, vgg_mean_pixel)

            avg_dist_AP += sess.run(batch_avg_dist_AP, feed_dict={anchor_image : anchor, positive_image: positive})
            avg_dist_AN += sess.run(batch_avg_dist_AN, feed_dict={anchor_image : anchor, negative_image: negative})
            
            
        avg_dist_AP = avg_dist_AP
        avg_dist_AN = avg_dist_AN

        print('Average distance A-P: %e' % avg_dist_AP)
        print('Average distance A-N: %e' % avg_dist_AN)

        # Collect extra parameters to save in .mat-file
        extra_parameters = {}
        extra_parameters['avg_dist_AP'] = avg_dist_AP
        extra_parameters['avg_dist_AN'] = avg_dist_AN
        extra_parameters['mean_pixel'] = vgg_mean_pixel
        

        print('Saving parameters...')
        file_name = trained_vgg.save_parameters(sess, extra_parameters, save_file_name)
        print('Parameters saved in: ' + file_name + '.m')
        
            
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

