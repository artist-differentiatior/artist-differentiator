import os
import time
from shutil import copy2
from collections import OrderedDict
import csv

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from tqdm import tqdm # progressbar

from PIL import Image
import numpy as np
import tensorflow as tf

import trained_vgg

from load_images import *

#STYLE_LAYERS = ('relu1_1', 'relu2_1')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')
VGG_PATH = 'vgg_net_original.mat'
CSV_FILE_PATH = 'new_train_info.csv'
PCA_PATH = 'pca_images/'
PREPROCESSED_PATH = 'preprocessed_images/'
ORIGINAL_FILE_PATH = 'sample_images/'



def apply_pca(weight_path, preprocessed_path=PREPROCESSED_PATH):

    """
    Trains the neural net using triplet loss

    Args: 
        network: (str) filepath to pretrained network parameters
        epochs: (int) number of training epochs
        learning_rate: (float) learning rate
        beta1: (float) momentum parameter for adam optimizer
        beta2: (float) RMSprop parameter for adam optimizer
        epsilon: (float) prevent division with 0 in adam optimizer
        batch_size: (int) size of mini batches

    """

    if not os.path.isfile(weight_path):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % VGG_PATH)

    print('Copying images to: ' + PCA_PATH)
    pca_files_dict, n_paintings_dict = _create_dir_for_pca(preprocessed_path, PCA_PATH)
    artists = n_paintings_dict.keys()
    
    print('Building net...')
    parameter_dict = trained_vgg.load_net(weight_path)
    vgg_mean_pixel = parameter_dict['mean_pixel']

    image = tf.placeholder('float', shape=(None, 224,224,3))
    
    with tf.variable_scope("net", reuse=tf.AUTO_REUSE):
        image_net = trained_vgg.net_preloaded(parameter_dict, image)

    image_styles = _generate_style(image_net, STYLE_LAYERS)

    
    # List of the computed PCA's for artist 1 and artist 2
    gram_data = {}

    # Initialize image loader
    image_loader = Image_Loader(PCA_PATH, 1, load_size=1)

    #saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        artist_index = 0
        img_nr = 1
        print('Computing gram matrices...')
        for img in tqdm(image_loader):
            
            current_artist = artists[artist_index]
            if img_nr > n_paintings_dict[current_artist]: # If we have gone through all paintings for this artist
                img_nr = 1
                artist_index += 1
                current_artist = artists[artist_index]
            
            
            img = trained_vgg.preprocess(img, vgg_mean_pixel)

            gram_all_layers = []
            for layer in STYLE_LAYERS:
                gram = sess.run(image_styles[layer], feed_dict={image: img}) # Compute gram matrices
                gram = gram.reshape(gram.shape[0]*gram.shape[1]*gram.shape[2]) # Flatten gram matrices
                
            gram_all_layers = np.concatenate((gram_all_layers, gram), axis=0) # Concatanate with gram matrices of other layers

            if current_artist not in gram_data: # Add the gram data to the corresponding artists entry in dictionary
                gram_data[current_artist] = [gram_all_layers]
            else:
                gram_data[current_artist].append(gram_all_layers)
                

            img_nr += 1


    # Apply PCA
    pca = PCA(n_components=2)

    n_colors = len(gram_data.keys())
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
    
    plt.figure(1)

    i = 1
    print('Computing PCA...')
    for artist, gram in gram_data.iteritems():
        pca_out = np.array(pca.fit_transform(gram))
        pca_out = pca_out.T # Transpose to get shape (2, n_points) 
    
        plt.scatter(pca_out[0], pca_out[1], c=colors[i-1], label='Artist ' + str(i))

        i += 1
    plt.legend()
    plt.show()

    
            
def _generate_style(net, style_layers):
    styles = {}

    for layer in style_layers:
        features = net[layer]
        features = tf.reshape(features, [-1, features.shape[1] * features.shape[2], features.shape[3]])
        gram = tf.divide(tf.matmul(tf.transpose(features, perm=[0,2,1]), features), tf.size(features[0], out_type=tf.float32))
        styles[layer] = gram

    return styles

def _create_dir_for_pca(preprocessed_path, pca_path):
    """
    Parses csv-file and creates directory with all preprocessed images in order of artist
    """

    filename_dict, n_paintings = _parse_info_file(CSV_FILE_PATH, ORIGINAL_FILE_PATH)

    pca_files_dict = {}
    n_paintings_dict = {}
    files_covered = []
    
    total_nr_paintings = 0
    for artist, paintings_by_artist in filename_dict.iteritems():
        total_nr_paintings += len(paintings_by_artist)
        n_paintings_dict[artist] = len(paintings_by_artist)
        
    i = 1
    for artist, paintings_by_artist in filename_dict.iteritems(): # Iterate over all artists

        for painting_name in paintings_by_artist: # Iterate over this artists paintings

            for file in os.listdir(preprocessed_path): # Go through the preprocessed files

                if painting_name in files_covered:
                    break

                if ('-' + painting_name) in file: # Find the corresponding preprocessed painting
                    
                    if len(str(i)) < len(str(total_nr_paintings)):
                        prefix = (len(str(total_nr_paintings)) - len(str(i)))*'0' + str(i)
                    else:
                        prefix = str(i)

                    new_filename = prefix + '_' + painting_name
                    src = preprocessed_path + file
                    dest = pca_path + new_filename
                    
                    copy2(src, dest) # copy the file to new dir
                        
                    if artist not in pca_files_dict: # Add this file to the pca files 
                        pca_files_dict[artist] = [new_filename]
                    else:
                        pca_files_dict[artist].append(new_filename)
                        
                    i += 1
                    files_covered.append(painting_name)

            
    return pca_files_dict, n_paintings_dict

    
        
def _parse_info_file(csv_file_path, original_paintings_file_path):

    '''
    Parses info file. Creates dictionary with keyword as artists and the corresponding value an 
    array of painting file-names.

    Args:
        csv_file_path: (str) path to csv file containing info about dataset
    '''
    file_names = []
    
    file_names = os.listdir(original_paintings_file_path)

    with open(csv_file_path, "r") as info_file:

        info_reader = csv.reader(info_file, dialect="excel", delimiter=",", quotechar="\"")

        artist_dict = {} # key=artist, value=[file_name, ...]
        info = info_reader.next()
        
        artist_index = info.index("artist")
        filename_index = info.index("filename")

        n_paintings = 0
        for row in info_reader:
            
            artist_name = row[artist_index]
            file_name = row[filename_index]

            if file_name not in file_names:
                continue
            
            if artist_name not in artist_dict:
                artist_dict[artist_name] = [file_name]
            else:
                artist_dict[artist_name].append(file_name)

            n_paintings = n_paintings + 1

        
    return artist_dict, n_paintings
    

if __name__ == "__main__":

    if len(sys.argv)==3:
        apply_pca(sys.argv[1], sys.argv[2])
    else:
        apply_pca(sys.argv[1])
