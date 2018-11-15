import numpy as np
import math
import sys

from scipy.misc import imread
from os import listdir

class Image_Loader:

    def __init__(self, image_source, mini_batch_size=3, triplet=True):
        self.__image_source = image_source

        #3 if triplets should be loaded, 2 otherwise(image pair)
        self.__load_size = 3 if triplet else 2

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
                #anchor, positive, negative
                return images[0::self.__load_size], images[2::self.__load_size], images[1::self.__load_size]

            return images[0::self.__load_size], images[1::self.__load_size]
        
        else:
            raise StopIteration

    def __len__(self):
        return self.__num_mini_batches


if __name__ == "__main__":

    print(sys.argv[1])
    image_loader = Image_Loader(sys.argv[1], triplet=False)
    

    for mini_batch in image_loader:
        anchors, positives = mini_batch



        

