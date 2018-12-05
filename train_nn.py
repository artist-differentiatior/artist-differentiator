import os
import time
import csv
import datetime
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import tensorflow as tf

import trained_vgg

from load_images import *
import logging


NAME_STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
PREPROCESSED_PATH = './train_data/'

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
        anchor_net = trained_vgg.net_preloaded(parameter_dict, anchor_image, 0)
        positive_net = trained_vgg.net_preloaded(parameter_dict, positive_image, 0)
        negative_net = trained_vgg.net_preloaded(parameter_dict, negative_image, 0)

    anchor_styles = _generate_style(anchor_net, style_layers)
    positive_styles = _generate_style(positive_net, style_layers)
    negative_styles = _generate_style(negative_net, style_layers)

    loss_sum = tf.add_n([tf.maximum(tf.reduce_sum(positive_weight*(anchor_styles[layer] - positive_styles[layer])**2 -
                                                  (anchor_styles[layer] - negative_styles[layer])**2, [1,2]) + loss_threshold, 0) for layer in style_layers])

 
    loss = tf.reduce_sum(loss_sum) / tf.to_float(tf.shape(loss_sum)[0])
    
    # Used to compute threshold for evaluating on pairs of images
    dist_p = tf.add_n([tf.reduce_sum((anchor_styles[layer] - positive_styles[layer]) ** 2,[1,2]) for layer in style_layers])
    dist_n = tf.add_n([tf.reduce_sum((anchor_styles[layer] - negative_styles[layer]) ** 2,[1,2]) for layer in style_layers])
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
    logging.info(" ------------ Options ------------")
    logging.info("Network: %s" % network)
    logging.info("Epochs: %d" % epochs)
    logging.info("Learning rate: %e" % learning_rate)
    logging.info("Mini batch size: %d" % batch_size)
    logging.info("Weight destination: %s" % save_file_name)
    logging.info("Loss threshold: %e" % loss_threshold)
    logging.info("Style layers: %s" % style_layers_indices)
    logging.info("Using checkpoints: %s" % checkpoints)
    logging.info("Device: %s" % device_name)
    logging.info("------------------------------------------")
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
        cost_data = []
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

                cost_data.append(cost)

                iteration_num += 1
                logging.info("Epoch: " + str(i + 1) + "/" + str(epochs) + \
                             " Iteration: " + str(iteration_num) + "/" + str(len(image_loader)) + \
                             " Cost: " + str(cost))


            save_path = saver.save(sess, "checkpoints/model.ckpt")
            plt.plot(cost_data)
            plt.savefig('cost_fig.pdf')
          
            epoch_end = time.time()
            epoch_times.append(epoch_end - epoch_start)

        print('Training completed. Computing mean distances...')

        # Get file names
        image_file_names = os.listdir(PREPROCESSED_PATH)
        image_file_names.sort()
        
        #sum_avg_dist_AP = 0
        #sum_avg_dist_AN = 0
        count = 0
        gram_matrix_dict = {}
        artist_dict = _parse_info_file('train_info.csv', PREPROCESSED_PATH)
        artist_list = _convert_dict_to_list(artist_dict)

        single_image_loader = Image_Loader(PREPROCESSED_PATH, 1)
    
        for anchor, positive, negative in tqdm(single_image_loader):

            anchor_file_name = image_file_names[count]
            painting_name = anchor_file_name.split('-')[1]

            index = np.where(artist_list[:,1] == painting_name)
            artist = np.asscalar(artist_list[index,0])
            
            anchor = trained_vgg.preprocess(anchor, vgg_mean_pixel)
            #positive = trained_vgg.preprocess(positive, vgg_mean_pixel)
            #negative = trained_vgg.preprocess(negative, vgg_mean_pixel)

            gram_matrix = sess.run(anchor_styles['relu5_1'], feed_dict={anchor_image: anchor})

            if artist in gram_matrix_dict.keys():
                gram_matrix_dict[artist] = np.array((gram_matrix_dict[artist] + gram_matrix)/2).astype(np.float)
            else:
                gram_matrix_dict[artist] = np.array(gram_matrix).astype(np.float)
            
            count += 3
            
            #sum_avg_dist_AP += sess.run(batch_avg_dist_AP, feed_dict={anchor_image : anchor, positive_image: positive})
            #sum_avg_dist_AN += sess.run(batch_avg_dist_AN, feed_dict={anchor_image : anchor, negative_image: negative})
            
            
        #avg_dist_AP = sum_avg_dist_AP / len(image_loader)
        #avg_dist_AN = sum_avg_dist_AN / len(image_loader)

        #print('Average distance A-P: %e' % avg_dist_AP)
        #print('Average distance A-N: %e' % avg_dist_AN)

        # Collect extra parameters to save in .mat-file
        extra_parameters = {}
        #extra_parameters['avg_dist_AP'] = avg_dist_AP
        #extra_parameters['avg_dist_AN'] = avg_dist_AN
        
        extra_parameters['mean_pixel'] = vgg_mean_pixel
        extra_parameters.update(gram_matrix_dict)

        print('Saving parameters...')
        file_name = trained_vgg.save_parameters(sess, extra_parameters, save_file_name)
        print('Parameters saved in: ' + file_name + '.mat')
        
            
def _generate_style(net, style_layers):
    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[1] * features.shape[2], features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features, perm=[0,2,1]), features), tf.size(features[0], out_type=tf.float32))
        styles[layer] = gram

    return styles

def _convert_dict_to_list(artist_dict):

    new_list = []
    for artist, paintings in artist_dict.iteritems():
        for painting in paintings:
            new_list.append([artist, painting])

    return np.array(new_list)

def _parse_info_file(csv_file_path, paintings_file_path):

    '''
    Parses info file. Creates dictionary with keyword as artists and the corresponding value as an 
    array of paintings.

    Args:
        csv_file_path: (str) path to csv file containing info about dataset
        paintings_file_path: (str) path to directory containing dataset
    '''
    file_names = []
    
    file_names = os.listdir(paintings_file_path)
    for i in range(len(file_names)):
        file_names[i] = file_names[i].split('-')[1]

    

    with open(csv_file_path, "r") as info_file:

        info_reader = csv.reader(info_file, dialect="excel", delimiter=",", quotechar="\"")

        artist_dict = {} # key=artist, value=[[file_name, style], ...]
        info = info_reader.next()
        
        artist_index = info.index("artist")
        filename_index = info.index("filename")

        for row in info_reader:
            
            artist_name = row[artist_index]
            file_name = row[filename_index]

            if file_name not in file_names:
                continue
            
            
            artist_dict_info = file_name
            
            if artist_name not in artist_dict:
                artist_dict[artist_name] = [artist_dict_info]
            else:
                artist_dict[artist_name].append(artist_dict_info)

        
    return artist_dict

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

