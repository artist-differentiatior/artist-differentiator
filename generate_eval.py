from random import shuffle
import csv
import os
import sys

from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from keras.preprocessing.image import load_img

IMGS_DIM_2D = (224, 224)
PREPROCESSED_IMAGE_DIR = "gen_test"

def load_and_save(img_name1, img_name2, num):

    img1 = load_img(sys.argv[1] + img_name1)
    img2 = load_img(sys.argv[1] + img_name2)

    img1 = fit(img1, IMGS_DIM_2D, method=LANCZOS)
    img2 = fit(img2, IMGS_DIM_2D, method=LANCZOS)

    if num < 10:
        num = '0' + str(num)

    img1.save(PREPROCESSED_IMAGE_DIR + "/" + str(num) + "_A-" + img_name1)
    img2.save(PREPROCESSED_IMAGE_DIR + "/" + str(num) + "_B-" + img_name2)

file_names = os.listdir(sys.argv[1])

with open(sys.argv[2], "r") as info_file:

    info_reader = csv.reader(info_file, dialect="excel", delimiter=",", quotechar="\"")

    artist_dict = {} # key=artist, value=[[file_name, style], ...]
    style_dict = {} # key=style, value=[[file_name, artist], ...]
    info = info_reader.next()

    artist_index = info.index("artist")
    filename_index = info.index("filename")

    image_list = []

    for row in info_reader:
        
        artist_name = row[artist_index]
        file_name = row[filename_index]

        if file_name in file_names:
            image_list.append([file_name, artist_name])

shuffle(image_list)

num_same = 0
num_diff = 0

half = len(image_list) / 2

first_half = image_list[0:half]
second_half = image_list[half:]

log = []


for count, e1 in enumerate(first_half):

    if log.count(1) <= log.count(0):
        
        found = False
        for e2 in second_half:
            if e1[1] == e2[1]:
                log.append(1)
                second_half.remove(e2)
                load_and_save(e1[0],e2[0],count)
                found = True
                break
            
        if not found:
            log.append(0)
            e2 = second_half.pop(0)
            load_and_save(e1[0],e2[0],count)

    else:

        e2 = second_half.pop(0)

        if e1[1] != e2[1]:
            log.append(0)
        else:
            log.append(1)
        load_and_save(e1[0],e2[0],count)

print(log)
print(log.count(1))
print(log.count(0))





