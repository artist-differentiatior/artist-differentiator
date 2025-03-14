import os
import sys
import shutil
import random
import util
import math

from argparse import ArgumentParser
from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from PIL import Image
from PIL import ImageFile
from keras.preprocessing.image import load_img
from tqdm import tqdm

IMGS_DIM_2D = (224, 224)
TRAIN_DIR = "train_data"
DEV_DIR = "dev_data"
TEST_DIR = "test_data"

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_parser():

    """
    Build an input parser
    """

    parser = ArgumentParser()
    parser.add_argument('--source',
        dest='source', help='Source folder with images to use as data',
            metavar='SOURCE', required=True)
    parser.add_argument('--info-file',
        dest='info_file', help='CSV-file containing information about images',
            metavar='INFO_FILE', required=True)
    parser.add_argument('--num-anchors', type=int,
        dest='num_anchors', help='Number of anchors for each image',
            metavar='NUM_ANCHORS', default=3)
    parser.add_argument('--test-ratio', type=float,
        dest='test_ratio', help='Ratio of test data. If 0 then no test data will be created',
            metavar='TEST_RATIO', default=0)
    parser.add_argument('--dev-ratio', type=float,
        dest='dev_ratio', help='Ratio of dev data. If 0 then no dev data will be created',
            metavar='DEV_RATIO', default=0)

    return parser


def preprocess_data(source, info_file, num_anchors, test_ratio, dev_ratio):

    """
    Copies images from the source folder and rescale them to size 224x224. These
    images are then grouped into train, dev and test sets and these sets are save to the folders: train_data, dev_data and test_data

    Args: 
        source: (str) path to source directory
        info_file: (str) path to csv file contatining information about images in source directory
        num_anchors: (int) Number of anchors to create for each painting
        test_ratio: (float) Percentage of data to classify as test data
        dev_ratio: (float) Percentage of data to classify as dev data
    """

    if dev_ratio != 0:
         _create_directory(DEV_DIR)
    if test_ratio != 0:
         _create_directory(TEST_DIR)
    if test_ratio + dev_ratio < 1:
         _create_directory(TRAIN_DIR)
    
    train_dict = util.parse_info_file(info_file, source)

    if dev_ratio != 0 or test_ratio != 0:

        length_dict = sum([len(value) for key, value in train_dict.items()])

        if dev_ratio != 0:
            num_paintings = int(math.ceil(length_dict * dev_ratio))
            dev_dict, train_dict = pick_n_from_dict(train_dict, num_paintings)
            touple_array, answer = util.generate_touple(dev_dict, 1)
            _preprocess_images(source, touple_array, DEV_DIR, 2)
            _write("dev_answer", str(answer))
            print("dev_data complete!")

        if test_ratio != 0:
            num_paintings = int(math.ceil(length_dict * test_ratio))
            test_dict, train_dict = pick_n_from_dict(train_dict, num_paintings)         
            touple_array, answer = util.generate_touple(test_dict, 1)
            _preprocess_images(source, touple_array, TEST_DIR, 2)
            _write("test_answer", str(answer))
            print("test_data complete!")

    if test_ratio + dev_ratio < 1:
        triplets_array = util.generate_triplets(train_dict, num_anchors)
        _preprocess_images(source, triplets_array, TRAIN_DIR, 3)
        print("train_data complete!")



def _write(filename, text):

    """
    Write text to a txt file

    Args:
        filename: (string) Name of file to write to
        text: (string) Text to write to the file
    """

    file = open(filename + ".txt","w") 
    file.write(text) 
    file.close() 
        
        
def pick_n_from_dict(dictionary, num):

    """

    Takes a dictionary containing lists of paintings. The function
    then move "num" of the paintings into a new dictionary and remove 
    them from the old.

    Args:
        dictionary: (dict) Dictinary to remove paintings from
        num: (int) Number of paintings to remove

    """

    sub_dict = {}

    for i in range(num):

        keys = list(dictionary.keys())
        random_key = random.choice(keys)
        random_value = random.choice(dictionary[random_key])

        dictionary[random_key].remove(random_value)

        if len(dictionary[random_key]) == 0:
            del dictionary[random_key]

        if random_key in sub_dict:
            sub_dict[random_key].append(random_value)
        else:
            sub_dict[random_key] = [random_value]

    return sub_dict, dictionary


def _create_directory(directory):

    """
    Create a directory. Checks if directory already exist and prompt the user
    if he wants to replace it.

    Args:
        directory: (string) Name of directory to create
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))

    if os.path.exists(directory):

        buffer = raw_input("The folder: " + dir_path + "/" + directory + " already exist." + \
        "To proceed that folder needs to be removed. Do you want to remove it now? (Y/N):\n")

        if buffer != "Y" and buffer != "y":
            print("Exit")
            exit()

        shutil.rmtree(directory)
        print("The folder: " + dir_path + "/" + directory + " sucessfully removed.")
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(directory)
    print("Create folder: " + dir_path + "/" + directory)


def _preprocess_images(source, image_array, save_path, num_images):

    """
    Iterated through image_array and copies touples/triplets into: save_path
    

    Args: 
        source: (str) path to source directory
        image_array: (array) determine how triplets are grouped together
        save_path: (str) path to directory to save images in
        num_images: (int) number of images for each element in image array (3 if triplets, 2 if touples)
    """

    random_indices = range(len(image_array))

    if num_images == 3:
        random.shuffle(random_indices)

    total_nr_images = len(image_array)    

    counter = 1

    for index in tqdm(random_indices):
	
	num = '0'*(len(str(total_nr_images)) - len(str(counter))) + str(counter)
        images = image_array[index]
        _create_preprocessed_triplet(source, num, images, save_path, num_images)
        counter += 1


def _create_preprocessed_triplet(source, num, images, save_path, num_images):
    """
    Copies and resizes a triplet/touple from source folder

    Args: 
        source: (str) path to source directory
        num: (str) triplet/touple number
        images: (array) determine which files in source directory to use as triplet
        save_path: (str) path to directory to save images in
        num_images: (int) number of images in images to copy
    """

    assert num_images == 3 or num_images == 2

    #For triplet: Anchor, Positive, Negative
    TRIPLET_NAMES = ["_A-", "_P-", "_N-"]
    #For touple: Image A and B
    TOUPLE_NAMES = ["_A-", "_B-"]

    for i in range(num_images):
        image = load_img(source + images[i])
        image = fit(image, IMGS_DIM_2D, method=LANCZOS)
        if num_images == 3:
            image.save(save_path + "/" + str(num) + TRIPLET_NAMES[i] + images[i])
        else:
            image.save(save_path + "/" + str(num) + TOUPLE_NAMES[i] + images[i])


def main():

    """

    Start preprocessing data
    
    """

    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.info_file):
        parser.error("Info file %s does not exist.)" % options.info_file)
    if not os.path.isdir(options.source):
        parser.error("The source directory %s does not exist.)" % options.source)

    if options.dev_ratio + options.test_ratio > 1:
        raise ValueError('Invalid test/dev/train data ratios. Must sum to 1.')
    if options.dev_ratio + options.test_ratio == 1:
        buffer = raw_input("With the current data ratios, no training data will be created. Proceed? (Y/N):\n")
        if buffer != "Y" and buffer != "y":
            print("Exit")
            exit()
        
    

    preprocess_data(source=options.source,
                    info_file=options.info_file,
                    num_anchors=options.num_anchors,
                    test_ratio=options.test_ratio,
                    dev_ratio=options.dev_ratio
    )
  
if __name__ == "__main__":
    main()
