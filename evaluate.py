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

    """
    Build an input parser
    """

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

    Evaluate the weights that has been trained.

    Args:
        test_path: (string) Path to directory with test images
        weight_path: (string) Path to the trained weights
        style_layers_indices: (list) List of which layers to extract style from (should 
                                     match the layers the weights has been trained on)
        data_type: (string) If data is test, dev or train

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

    vgg_mean_pixel = parameter_dict['mean_pixel']
    del parameter_dict['mean_pixel']
    
    gram_matrix_dict = parameter_dict

    prediction = []

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

                img1_gram = sess.run(image_1_styles['relu5_1'], feed_dict={image_1:img1})
                img2_gram = sess.run(image_1_styles['relu5_1'], feed_dict={image_1:img2})

                closest_artist_1 = _find_closest_artist(gram_matrix_dict, img1_gram)
                closest_artist_2 = _find_closest_artist(gram_matrix_dict, img2_gram)

                if closest_artist_1 == closest_artist_2:
                    prediction.append(1)
                else:
                    prediction.append(0)


        f1_accuracy = f1_score(answer, prediction)
        auc_score = roc_auc_score(answer, prediction)
       
        print('Answer    : ' + str(answer))
        print('Prediction: ' + str(prediction))
        print('F1 Score: %f' % f1_accuracy)
        print('AUC Score: %f' % auc_score)

            
def _generate_style(net, style_layers):

    """
    Generate a dictionary with gram matrices of the style layers specified in
    the parameter style_layers

    Args:
        net: (dict) Dictionary containing the neural net
        style_layers: (list) List specifying which layers to extract gram matrices from
    """

    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[1] * features.shape[2], features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features, perm=[0,2,1]), features), tf.size(features[0], out_type=tf.float32))
        styles[layer] = gram

    return styles


def _find_closest_artist(gram_matrix_dict, gram):

    """

    Compute which of the gram matrices in "gram_matrix_dict" the "gram" matrix is closest to.

    Args:
        gram_matrix_dict: (dict) Dictionary containing gram matrices corresponding to an artist
        gram: (numpy array) Gram matrix to be evaluated

    """

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

    """

    Start evaluating the weights

    """

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

