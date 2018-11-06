import numpy as np
import cv2
import math
import sys

from scipy.misc import imread
from os import listdir

class Image_Loader(object):

    def __init__(self, image_source, mini_batch_size=3):
        self.__image_source = image_source
        try:
            self.__images = listdir(self.__image_source)
        except Exception as e:
            print(e)
        self.__mini_batch = 0
        self.__mini_batch_size = mini_batch_size
        self.__num_mini_batches = int(math.ceil(len(self.__images) / float(self.__mini_batch_size)))

    def load_next_batch(self):

        start_im = self.__mini_batch * self.__mini_batch_size
        end_im = min((self.__mini_batch + 1) * self.__mini_batch_size, len(self.__images))
        print(start_im, end_im)

        images = []

        for i in range(start_im, end_im):
            filename = self.__image_source + "/" + self.__images[i]
            #image = imread(filename).astype(np.float)
            image = cv2.imread(filename)
            images.append(image)

        self.__mini_batch = (self.__mini_batch + 1) % self.__num_mini_batches

        return images


if __name__ == "__main__":

    image_loader = Image_Loader(sys.argv[1])

    while True:

        x = raw_input("Please enter an input: ")
        images = image_loader.load_next_batch()
        for img in images:
            cv2.imshow("hej", img)
            cv2.waitKey(300)