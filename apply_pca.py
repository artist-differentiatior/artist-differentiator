import os
import time
from shutil import copy2
from collections import OrderedDict
import csv
from argparse import ArgumentParser

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from tqdm import tqdm # progressbar

from PIL import Image
import numpy as np
import tensorflow as tf

import trained_vgg

from load_images import *
from util import parse_info_file_triplets, convert_dict_to_list

NAME_STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']

CSV_FILE_PATH = 'train_info.csv'
PREPROCESSED_PATH = 'preprocessed_images/'
VGG_PATH = 'vgg_net_original.mat'


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--weights',
            dest='weights', help='path to network weights (.mat file) (default %(default)s)',
                        metavar='WEIGHTS', default=VGG_PATH)
    parser.add_argument('--csv',
            dest='csv_file_path', help='path to csv-file (default %(default)s)',
                        metavar='VGG_PATH', default=CSV_FILE_PATH)
    parser.add_argument('--source',
            dest='preprocessed_path', help='path to preprocessed images to apply pca on (default %(default)s)',
                        metavar='PREPROCESSED_PATH', default=PREPROCESSED_PATH)
    parser.add_argument('--type',
                        dest='type', help='type of data (train, dev or test) (default %(default)s)',
                        metavar='type', default='train')
    parser.add_argument('--style-layers-indices', nargs='+', type=int,
            dest='style_layers_indices', help='Which layers to use(0-4)',
                        metavar='STYLE_LAYERS_INDICES', default=[3,4])

    return parser


def apply_pca(weight_path, csv_file_path, preprocessed_path, data_type, style_layers_indices):

    """
    Computes pca for the supplied data and outputs figure pca.pdf

    Args: 
        weight_path: (str) file path to pretrained network parameters (.mat file)
        csv_file_path: (str) file path to data info file (.csv file)
        preprocessed_path: (str) file path to preprocessed data to apply pca on
        data_type: (str) the type of data supplied (train, dev or test)
        style_layers_indices: (int array) indices to NAME_STYLE_LAYERS for which layers to compute on

    """

    style_layers = [NAME_STYLE_LAYERS[i] for i in style_layers_indices]

    if not os.path.isfile(weight_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % VGG_PATH)
    
    print('Building net...')
    parameter_dict = trained_vgg.load_net(weight_path)
    vgg_mean_pixel = parameter_dict['mean_pixel']

    image = tf.placeholder('float', shape=(None, 224,224,3))
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        image_net = trained_vgg.net_preloaded(parameter_dict, image)

    image_styles = _generate_style(image_net, style_layers)

    gram_dict = {}
    all_grams = []

    # Initialize image loader
    if data_type == 'train':
        image_loader = Image_Loader(preprocessed_path, 1, load_size=3)
    else:
        image_loader = Image_Loader(preprocessed_path, 1, load_size=2)

    # Get dictionary of with all paintings in preprocessed_path, get list of file names
    artist_dict = parse_info_file_triplets(csv_file_path, preprocessed_path)
    image_file_names = os.listdir(preprocessed_path)
    image_file_names.sort()
    artist_list = convert_dict_to_list(artist_dict)
    done_image_names = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())


        if data_type == 'train': # for train data
        
            count = 0
            print('Computing gram matrices...')
            for img, throw_away1, throw_away2 in tqdm(image_loader):

                img_file_name = image_file_names[count]
                painting_name = img_file_name.split('-')[1]

                count += 3
                
		if painting_name in done_image_names:
		    continue
		done_image_names.append(painting_name)

                index = np.where(artist_list[:,1] == painting_name)
                artist = np.asscalar(artist_list[index,0])
            
                img = trained_vgg.preprocess(img, vgg_mean_pixel)

                gram_all_layers = []
                for layer in style_layers:
                    gram = sess.run(image_styles[layer], feed_dict={image: img}) # Compute gram matrices
                    gram = gram.reshape(gram.shape[0]*gram.shape[1]*gram.shape[2]) # Flatten gram matrices
                    gram_all_layers = np.concatenate((gram_all_layers, gram), axis=0) # Concatanate with gram matrices of other layers

                if artist not in gram_dict: # Add the gram data to the corresponding artists entry in dictionary
                    gram_dict[artist] = [gram_all_layers]
                else:
                    gram_dict[artist].append(gram_all_layers)

                all_grams.append(gram_all_layers)
                

                
                
        else: # for test/dev data

            count = 0
            print('Computing gram matrices...')
            for img, throw_away1 in tqdm(image_loader):

                img_file_name = image_file_names[count]
                painting_name = img_file_name.split('-')[1]

                count += 2

		if painting_name in done_image_names:
		    continue
		done_image_names.append(painting_name)

                index = np.where(artist_list[:,1] == painting_name)
                artist = np.asscalar(artist_list[index,0])
            
                img = trained_vgg.preprocess(img, vgg_mean_pixel)

                gram_all_layers = []
                for layer in style_layers:
                    gram = sess.run(image_styles[layer], feed_dict={image: img}) # Compute gram matrices
                    gram = gram.reshape(gram.shape[0]*gram.shape[1]*gram.shape[2]) # Flatten gram matrices
                    gram_all_layers = np.concatenate((gram_all_layers, gram), axis=0) # Concatanate with gram matrices of other layers

                if artist not in gram_dict: # Add the gram data to the corresponding artists entry in dictionary
                    gram_dict[artist] = [gram_all_layers]
                else:
                    gram_dict[artist].append(gram_all_layers)

                all_grams.append(gram_all_layers)
                

    # Compute PCA
    scaler = StandardScaler()
    pca = PCA(n_components=2)

    print('Computing PCA...')
    scaler.fit(all_grams)
    standard_all_grams = scaler.transform(all_grams)
    pca.fit(standard_all_grams)

    n_colors = len(gram_dict.keys())
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]

    plt.figure(1)

    i = 1
    print('Applying PCA...')
    for artist, gram in tqdm(gram_dict.iteritems()):
        standard_gram = scaler.transform(gram)
        pca_out = np.array(pca.transform(standard_gram))
        pca_out = pca_out.T # Transpose to get shape (2, 1) 
    
        plt.scatter(pca_out[0], pca_out[1], c=colors[i-1], label='Artist ' + str(i))

        i += 1
        
    plt.legend()

    plt.savefig('pca.pdf')

    
            
def _generate_style(net, style_layers):
    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[1] * features.shape[2], features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features, perm=[0,2,1]), features), tf.size(features[0], out_type=tf.float32))
        styles[layer] = gram

    return styles


if __name__ == "__main__":

    parser = build_parser()
    options = parser.parse_args()

    if options.type not in ['train', 'dev', 'test']:
        raise ValueError('Invalid type of data. Choose test, dev or train')
    
    apply_pca(
        options.weights,
        options.csv_file_path,
        options.preprocessed_path,
        options.type,
        options.style_layers_indices
    
    )
