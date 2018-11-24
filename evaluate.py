import os
import time
import numpy as np
import tensorflow as tf

import trained_vgg

from load_images import Image_Loader
from collections import OrderedDict
from argparse import ArgumentParser
from tqdm import tqdm
from PIL import Image

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
    parser.add_argument('--style-layers-indices', type=list,
        dest='style_layers_indices', help='Which layers to use(0-4)',
            metavar='STYLE_LAYERS_INDICES', default=[3,4])
    parser.add_argument('--triplet', type=bool,
        dest='triplet', help='If triplet or touple',
            metavar='TRIPLET', default=False)

    return parser


def evaluate(test_path, weight_path, style_layers_indices, triplet):

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
    vgg_mean_pixel = parameter_dict['mean_pixel']

    image_1 = tf.placeholder('float', shape=(None, 224,224,3))
    image_2 = tf.placeholder('float', shape=(None, 224,224,3))
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        image_1_net = trained_vgg.net_preloaded(parameter_dict, image_1)
        image_2_net = trained_vgg.net_preloaded(parameter_dict, image_2)

    image_1_styles = _generate_style(image_1_net, style_layers)
    image_2_styles = _generate_style(image_2_net, style_layers)

    compute_dist = tf.add_n([tf.reduce_sum((image_1_styles[layer] - image_2_styles[layer]) ** 2,[1,2]) for layer in style_layers])

    
    # Initialize image loader
    if not triplet:
        image_loader = Image_Loader(test_path, 1, load_size=2)
    else:
        image_loader = Image_Loader(test_path, 1, load_size=3)

    avg_dist_AP = parameter_dict['avg_dist_AP']
    avg_dist_AN = parameter_dict['avg_dist_AN']
    print('Average distance AP: %e' % avg_dist_AP)
    print('Average distance AN: %e'% avg_dist_AN)
    harmonic_mean_threshold = 2*avg_dist_AP*avg_dist_AN/(avg_dist_AP + avg_dist_AN)

    prediction = []

    #saver = tf.train.Saver()

    print('Harmonic mean:', harmonic_mean_threshold)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        if triplet:

            for img1, img2, img3 in image_loader:

                img1 = trained_vgg.preprocess(img1, vgg_mean_pixel)
                img2 = trained_vgg.preprocess(img2, vgg_mean_pixel)
                img3 = trained_vgg.preprocess(img2, vgg_mean_pixel)

                dist1 = sess.run(compute_dist, feed_dict={image_1 : img1, image_2: img2})

                if dist1 <= harmonic_mean_threshold:
                    prediction.append(1)
                else:
                    prediction.append(0)

                print(dist1)

                dist2 = sess.run(compute_dist, feed_dict={image_1 : img1, image_2: img3})
                
                if dist2 <= harmonic_mean_threshold:
                    prediction.append(1)
                else:
                    prediction.append(0)

                print(dist2)

        else:

            for img1, img2 in image_loader:

                img1 = trained_vgg.preprocess(img1, vgg_mean_pixel)
                img2 = trained_vgg.preprocess(img2, vgg_mean_pixel)

                dist = sess.run(compute_dist, feed_dict={image_1 : img1, image_2: img2})
                
                if dist <= harmonic_mean_threshold:
                    prediction.append(1)
                else:
                    prediction.append(0)

                print(dist)
            
        print('Prediction:' + str(prediction))

    
            
def _generate_style(net, style_layers):
    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[1] * features.shape[2], features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features, perm=[0,2,1]), features), tf.size(features[0], out_type=tf.float32))
        styles[layer] = gram

    return styles

def main():
    parser = build_parser()
    options = parser.parse_args()

    evaluate(test_path=options.test_path,
             weight_path=options.weight_path,
             style_layers_indices=options.style_layers_indices,
             triplet=options.triplet)


if __name__ == "__main__":
    main()

