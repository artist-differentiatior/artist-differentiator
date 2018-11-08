import os
import sys
import time
import shutil

from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from keras.preprocessing.image import load_img

IMGS_DIM_2D = (224, 224)
PREPROCESSED_IMAGE_DIR = "preprocessed_images"

def generate_preprocessed_images(source):
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
    _preprocesse_images(source)

def _preprocesse_images(source):
    source_dir = os.listdir(source)
    print("Create new images")

    count = 0
    output = "Images created %s/%s" %(count, len(source_dir))
    sys.stdout.write(output)
    sys.stdout.flush()

    for image_name in source_dir:
        _create_preprocessed_image(source, image_name)
        sys.stdout.write("\b" * len(output))
        count += 1
        output = "Images created %s/%s" %(count, len(source_dir))
        sys.stdout.write(output)
        sys.stdout.flush()
    print("\nFinished creating images")

def _create_preprocessed_image(source, name):
    image = load_img(source + name)
    image = fit(image, IMGS_DIM_2D, method=LANCZOS)
    image.save(PREPROCESSED_IMAGE_DIR + "/" + name)

if __name__ == "__main__":
    generate_preprocessed_images(sys.argv[1])