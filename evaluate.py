import os
import time
from collections import OrderedDict
#from tqdm import tqdm # progressbar

from PIL import Image
import numpy as np
import tensorflow as tf

import vgg

from load_images import *

STYLE_LAYERS = ('relu1_1', 'relu2_1')
#STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
VGG_PATH = 'imagenet-vgg-verydeep-19.mat'


def evaluate(test_path):

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

    if not os.path.isfile(VGG_PATH):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % VGG_PATH)

    vgg_weights, vgg_mean_pixel = vgg.load_net(VGG_PATH)

    image_1 = tf.placeholder('float', shape=(None, 224,224,3))
    image_2 = tf.placeholder('float', shape=(None, 224,224,3))
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        image_1_net = vgg.net_preloaded(vgg_weights, image_1)
        image_2_net = vgg.net_preloaded(vgg_weights, image_2)

    image_1_styles = _generate_style(image_1_net, STYLE_LAYERS)
    image_2_styles = _generate_style(image_2_net, STYLE_LAYERS)

    loss_threshold = 1000000

    compute_dist = tf.add_n([tf.reduce_sum((image_1_styles[layer] - image_2_styles[layer]) ** 2,[1,2]) for layer in STYLE_LAYERS])

    # Initialize image loader
    image_loader = Image_Loader(test_path, 1, triplet=False)

    #saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for img1, img2 in image_loader:
            dist = sess.run(compute_dist, feed_dict={image_1 : img1, image_2: img2})
            print(dist)

            
def _generate_style(net, style_layers):
    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[1] * features.shape[2], features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features, perm=[0,2,1]), features), tf.size(features[0], out_type=tf.float32))
        styles[layer] = gram

    return styles

if __name__ == "__main__":

    evaluate(sys.argv[1])