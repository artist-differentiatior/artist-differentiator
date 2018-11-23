import numpy as np
import math
import sys

from scipy.misc import imread
from os import listdir


class Image_Loader:

    """
    This class is an iterator and its purpose is to load batches with images from a folder.
    """

    def __init__(self, image_source, mini_batch_size=3, load_size=3):

        """

        Args: 
            image_source: (str) filepath to folder with images to load
            mini_batch_size: (int) amount of images that should be loaded each iteration
            load_size: (int) load mode: 3 for triplet, 2 for touple and 1 for regular loading

        """

        self.__image_source = image_source
        self.__load_size = load_size

        try:
            self.__images = listdir(self.__image_source)
            self.__images.sort()
        except Exception as e:
            print(e)

        self.__total_mini_batch_size = self.__load_size*mini_batch_size
        self.__num_mini_batches = int(math.ceil(len(self.__images) / float(self.__total_mini_batch_size)))
        


    def __iter__(self):

        self.__mini_batch_nr = 0
        self.__start_im = 0
        
        return self

    def next(self):
        
        if self.__mini_batch_nr < self.__num_mini_batches:

            self.__end_im = min(self.__start_im + self.__total_mini_batch_size, len(self.__images))

            images = []

            for i in range(self.__start_im, self.__end_im):

                image_filename = self.__image_source + "/" + self.__images[i]
                image = imread(image_filename).astype(np.float)
                images.append(image)

            self.__start_im += self.__total_mini_batch_size
            self.__mini_batch_nr = self.__mini_batch_nr + 1

            if self.__load_size == 3:
                # anchor, positive, negative
                return images[0::self.__load_size], images[2::self.__load_size], images[1::self.__load_size]
            elif self.__load_size == 2:
                # image1, image2
                return images[0::self.__load_size], images[1::self.__load_size]
            elif self.__load_size == 1:
                return images[0::self.__load_size]
            else:
                raise ValueError('Illegal value for image load size. Should be 1, 2 or 3')
        
        else:
            raise StopIteration

    def __len__(self):
        return self.__num_mini_batches



        

