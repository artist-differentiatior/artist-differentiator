import os
import time
import numpy as np
import tensorflow as tf
import ast

import trained_vgg

from load_images import Image_Loader
from collections import OrderedDict
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import f1_score, roc_auc_score

NAME_STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
VGG_PATH = 'vgg_net_original.mat'


def build_parser():

    parser = ArgumentParser()
    parser.add_argument('--test-path',
        dest='test_path', help='Path to folder with images to test on',
            metavar='TEST_PATH', required=True)
    parser.add_argument('--weight-path',
        dest='weight_path', help='Path to weights',
            metavar='WEIGHT_PATH', required=True)
    parser.add_argument('--style-layers-indices', nargs='+', type=int,
        dest='style_layers_indices', help='Which layers to use(0-4)',
            metavar='STYLE_LAYERS_INDICES', default=[3,4])
    parser.add_argument('--type', type=str,
                        dest='type', help='What type of data is evaluated on (train, dev or test)',
                        metavar='TYPE', default='test')

    return parser


def evaluate(test_path, weight_path, style_layers_indices, data_type):

    """
    Trains the neural net using triplet loss

    Args: 
        network: (str) filepath to pretrained network parameters
        epochs: (int) number of training epochs
        learning_rate: (float) learning rate
        beta1: (float) momentum parameter for adam optimizer
        beta2: (float) RMSprop parameter for adam optimizer
        epsilon: (float) prevent division with 0 in adaom optimizer
        batch_size: (int) size of mini batches

    """

    style_layers = [NAME_STYLE_LAYERS[i] for i in style_layers_indices]

    if not os.path.isfile(weight_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % VGG_PATH)

    parameter_dict = trained_vgg.load_net(weight_path)

    image_1 = tf.placeholder('float', shape=(None, 224,224,3))
    image_2 = tf.placeholder('float', shape=(None, 224,224,3))
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        image_1_net = trained_vgg.net_preloaded(parameter_dict, image_1)
        image_2_net = trained_vgg.net_preloaded(parameter_dict, image_2)

    image_1_styles = _generate_style(image_1_net, style_layers)
    image_2_styles = _generate_style(image_2_net, style_layers)

    compute_dist = tf.add_n([tf.reduce_sum((image_1_styles[layer] - image_2_styles[layer]) ** 2,[1,2]) for layer in style_layers])

    
    # Initialize image loader
    if data_type != 'train':
        image_loader = Image_Loader(test_path, 1, load_size=2)
    else:
        image_loader = Image_Loader(test_path, 1, load_size=3)

    #avg_dist_AP = parameter_dict['avg_dist_AP']
    #avg_dist_AN = parameter_dict['avg_dist_AN']

    #harmonic_mean_threshold = 2*avg_dist_AP*avg_dist_AN/(avg_dist_AP + avg_dist_AN)
    #harmonic_mean_threshold = (avg_dist_AP+avg_dist_AN)/2

    vgg_mean_pixel = parameter_dict['mean_pixel']
    del parameter_dict['mean_pixel']
    
    gram_matrix_dict = parameter_dict

    print(gram_matrix_dict.keys())


    prediction = []

    #saver = tf.train.Saver()

    with tf.Session() as sess:

        print('Computing distances...')

        sess.run(tf.global_variables_initializer())

        if data_type == 'train':

            answer = [1, 0] * len(image_loader)
	    
            for img1, img2, img3 in tqdm(image_loader):

                img1 = trained_vgg.preprocess(img1, vgg_mean_pixel)
                img2 = trained_vgg.preprocess(img2, vgg_mean_pixel)
                img3 = trained_vgg.preprocess(img3, vgg_mean_pixel)

                img1_gram = sess.run(image_1_styles['relu5_1'], feed_dict={image_1:img1})
                img2_gram = sess.run(image_1_styles['relu5_1'], feed_dict={image_1:img2})
                img3_gram = sess.run(image_1_styles['relu5_1'], feed_dict={image_1:img3})

                closest_artist_1 = _find_closest_artist(gram_matrix_dict, img1_gram)
                closest_artist_2 = _find_closest_artist(gram_matrix_dict, img2_gram)
                closest_artist_3 = _find_closest_artist(gram_matrix_dict, img3_gram)

                if closest_artist_1 == closest_artist_2:
                    prediction.append(1)
                else:
                    prediction.append(0)

                if closest_artist_1 == closest_artist_3:
                    prediction.append(1)
                else:
                    prediction.append(0)
                
                #dist1 = sess.run(compute_dist, feed_dict={image_1 : img1, image_2: img2})

                #if dist1 <= harmonic_mean_threshold:
                #    prediction.append(1)
                #else:
                #    prediction.append(0)

                #print(dist1)

                #dist2 = sess.run(compute_dist, feed_dict={image_1 : img1, image_2: img3})
                
                #if dist2 <= harmonic_mean_threshold:
                #    prediction.append(1)
                #else:
                #    prediction.append(0)

                #print(dist2)
		


        else:
            if data_type == 'dev':
                with open('dev_answer.txt', 'r') as dev_answer_file:
                    answer = ast.literal_eval(dev_answer_file.readline())
            elif data_type == 'test':
                with open('test_answer.txt', 'r') as test_answer_file:
                    answer = ast.literal_eval(test_answer_file.readline())
                
            

            for img1, img2 in tqdm(image_loader):

                img1 = trained_vgg.preprocess(img1, vgg_mean_pixel)
                img2 = trained_vgg.preprocess(img2, vgg_mean_pixel)

                #dist = sess.run(compute_dist, feed_dict={image_1 : img1, image_2: img2})
                
                #if dist <= harmonic_mean_threshold:
                #    prediction.append(1)
                #else:
                #    prediction.append(0)

                #print(dist)


        f1_accuracy = f1_score(answer, prediction)
        auc_score = roc_auc_score(answer, prediction)

        
        #print('Average distance AP: %e' % avg_dist_AP)
        #print('Average distance AN: %e'% avg_dist_AN)
        #print('Harmonic mean: %e' % harmonic_mean_threshold)        
        print('Answer    : ' + str(answer))
        print('Prediction: ' + str(prediction))
        print('F1 Score: %f' % f1_accuracy)
        print('AUC Score: %f' % auc_score)

    
            
def _generate_style(net, style_layers):
    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[1] * features.shape[2], features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features, perm=[0,2,1]), features), tf.size(features[0], out_type=tf.float32))
        styles[layer] = gram

    return styles

def _find_closest_artist(gram_matrix_dict, gram):

    min = 1e12
    for artist, average_gram in gram_matrix_dict.iteritems():
        if artist[0] == 'b' or artist[0] == 'w':
            continue

        norm_dist = np.linalg.norm(gram - average_gram.astype(np.float))
        if norm_dist < min:
            min = norm_dist
            closest_artist = artist
    
    return closest_artist

def main():
    parser = build_parser()
    options = parser.parse_args()
    
    if options.type not in ['train', 'dev', 'test']:
        raise ValueError('Invalid type of data. Choose test, dev or train')

    evaluate(test_path=options.test_path,
             weight_path=options.weight_path,
             style_layers_indices=options.style_layers_indices,
             data_type=options.type)


if __name__ == "__main__":
    main()

