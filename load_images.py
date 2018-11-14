import numpy as np
import math
import sys

from scipy.misc import imread
from os import listdir
#import cv2

class Image_Loader:

    def __init__(self, image_source, mini_batch_size=3):
        self.__image_source = image_source

        try:
            self.__images = listdir(self.__image_source)
            self.__images.sort()
        except Exception as e:
            print(e)
        self.__total_mini_batch_size = 3*mini_batch_size
        self.__num_mini_batches = int(math.ceil(len(self.__images) / float(self.__total_mini_batch_size)))
        


    def __iter__(self):

        self.__mini_batch_nr = 0
        self.__start_im = -self.__total_mini_batch_size
        self.__end_im = -min(self.__total_mini_batch_size, len(self.__images))
        
        return self

    def next(self):
        
        if self.__mini_batch_nr < self.__num_mini_batches:

            self.__start_im += self.__total_mini_batch_size
            self.__end_im = end_im = min(self.__start_im + self.__total_mini_batch_size, len(self.__images))
            
            anchors = []
            negatives = []
            positives = []

            for i in range(self.__start_im, self.__end_im, 3):
                anchor_filename = self.__image_source + "/" + self.__images[i]
                negative_filename = self.__image_source + "/" + self.__images[i+1]
                positive_filename = self.__image_source + "/" + self.__images[i+2]

                anchor_image = imread(anchor_filename).astype(np.float)
                negative_image = imread(negative_filename).astype(np.float)
                positive_image = imread(positive_filename).astype(np.float)

                anchors.append(anchor_image)
                negatives.append(negative_image)
                positives.append(positive_image)

            self.__mini_batch_nr = self.__mini_batch_nr + 1

            return anchors, positives, negatives
        
        else:
            raise StopIteration

    def __len__(self):
        return self.__num_mini_batches


if __name__ == "__main__":

    print(sys.argv[1])
    image_loader = Image_Loader(sys.argv[1])
    

    for mini_batch in image_loader:
        anchors, positives, negatives = mini_batch
        
        x = raw_input("Please enter an input: ")
        for img in anchors:
            cv2.imshow("hej", img)
            cv2.waitKey(300)
