import os
import sys
import time
import shutil

from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from keras.preprocessing.image import load_img
from generate_triplets import *

IMGS_DIM_2D = (224, 224)
PREPROCESSED_IMAGE_DIR = "preprocessed_images"


def generate_preprocessed_images(source, triplet_array):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(PREPROCESSED_IMAGE_DIR):
        buffer = raw_input("The folder: " + dir_path + "/" + PREPROCESSED_IMAGE_DIR + " already exist." + \
        "To proceed that folder needs to be removed. Do you want to remove it now? (Y/N):\n")
        if buffer != "Y" and buffer != "y":
            print("Exit")
            return
        shutil.rmtree(PREPROCESSED_IMAGE_DIR)
        print("The folder: " + dir_path + "/" + PREPROCESSED_IMAGE_DIR + " sucessfully removed.")
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.makedirs(PREPROCESSED_IMAGE_DIR)
    print("Create folder: " + dir_path + "/" + PREPROCESSED_IMAGE_DIR)
    _preprocesse_images(source, triplet_array)


def _preprocesse_images(source, triplet_array):

    output = "Triplets created: %s/%s" %(0, len(triplet_array))
    sys.stdout.write(output)
    sys.stdout.flush()

    for counter, triplet in enumerate(triplet_array):
        _create_preprocessed_triplet(source, counter + 1, triplet)

        #Write progress
        sys.stdout.write("\b" * len(output))
        output = "Triplets created: %s/%s" %(counter + 1, len(triplet_array))
        sys.stdout.write(output)
        sys.stdout.flush()

    print("\nFinished creating images")


def _create_preprocessed_triplet(source, num, triplet):
    anchor = load_img(source + triplet[0])
    positive = load_img(source + triplet[1])
    negative = load_img(source + triplet[2])

    anchor = fit(anchor, IMGS_DIM_2D, method=LANCZOS)
    positive = fit(positive, IMGS_DIM_2D, method=LANCZOS)
    negative = fit(negative, IMGS_DIM_2D, method=LANCZOS)

    if num < 10:
        num = '0' + str(num)

    anchor.save(PREPROCESSED_IMAGE_DIR + "/" + str(num) + "_A-" + triplet[0])
    positive.save(PREPROCESSED_IMAGE_DIR + "/" + str(num) + "_P-" + triplet[1])
    negative.save(PREPROCESSED_IMAGE_DIR + "/" + str(num) + "_N-" + triplet[2])


if __name__ == "__main__":
    generate_preprocessed_images(sys.argv[1], generate_triplets("new_train_info.csv", sys.argv[1]))
