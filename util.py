import csv
import os
import random
import numpy as np


def parse_info_file(csv_file_path, paintings_file_path):

    '''
    Parses info file. Creates dictionary with keyword as artists and the corresponding value as an 
    array of paintings.

    Args:
        csv_file_path: (str) path to csv file containing info about dataset
        paintings_file_path: (str) path to directory containing dataset
    '''
    file_names = []
    
    file_names = os.listdir(paintings_file_path)

    with open(csv_file_path, "r") as info_file:

        info_reader = csv.reader(info_file, dialect="excel", delimiter=",", quotechar="\"")

        artist_dict = {} # key=artist, value=[[file_name, style], ...]
        info = info_reader.next()
        
        artist_index = info.index("artist")
        filename_index = info.index("filename")

        for row in info_reader:
            
            artist_name = row[artist_index]
            file_name = row[filename_index]

            if file_name not in file_names:
                continue
            
            artist_dict_info = file_name
            
            if artist_name not in artist_dict:
                artist_dict[artist_name] = [artist_dict_info]
            else:
                artist_dict[artist_name].append(artist_dict_info)

        
    return artist_dict


def generate_triplets(artist_dict, num_anchors):

    '''
    Generates triplets from a dictonary with artists as keys with painted paintings as value

    Args:
        artist_dict: (dict) Key (string): artist, value (array) corresponding paintings
        num_anchors: (int) number of anchors for each painting
    '''

    triplet_array = []

    assert len(artist_dict) > 1, "Just 1 artist"

    # Iterate over all artists
    for artist, paintings_by_artist in artist_dict.iteritems():
        
        assert len(paintings_by_artist) > 1, "Just 1 painting from %r, can't generate triplet!" %artist
        
        # Iterate over paintings by current artist
        for anchor_index, painting in enumerate(paintings_by_artist):

            for i in range(num_anchors):

                positive_index = random.randint(0,len(paintings_by_artist) - 1)
                #Not same index as anchor
                while positive_index == anchor_index:
                    positive_index = random.randint(0,len(paintings_by_artist) - 1)

                positive_painting = paintings_by_artist[positive_index]

                #Get negative painting
                artists = list(artist_dict.keys())
                #remove current artist
                artists.remove(artist)

                random_artist = random.choice(artists)
                negative_painting = random.choice(artist_dict[random_artist])

                triplet_array.append([painting, positive_painting, negative_painting]) # Add the new triplet to array

    return triplet_array

def generate_touple(artist_dict, num):

    '''
    Generates triplets from a dictonary with artists as keys with painted paintings as value

    Args:
        artist_dict: (dict) Key (string): artist, value (array) corresponding paintings
        num: (int) number of times to iterate over same painting to create a touple
    '''

    artist_keys = artist_dict.keys()
    nr_paintings = 0
    for artist in artist_keys:
        nr_paintings += len(artist_dict[artist])

    touple_array = []
    answer = []

    for artist in artist_keys:
        for painting in artist_dict[artist]:
            for count in range(num):
                    
                
                if sum(answer) > 0.5*len(answer): # If more positives - pick a random negative

                    artist1 = artist
                    painting1 = painting
                
                    artist2 = artist1
                    while artist2 == artist1:
                        artist2 = artist_keys[random.randint(0, len(artist_keys)-1)] # random different artist

                    painting2 = artist_dict[artist2][random.randint(0, len(artist_dict[artist2])-1)] # random painting by that artist

                    while ([painting1, painting2] in touple_array) or ([painting2, painting1] in touple_array): # if we already have that touple - try again
                        
                        artist2 = artist1
                        while artist2 == artist1:
                            artist2 = artist_keys[random.randint(0, len(artist_keys)-1)] # random different artist

                        painting2 = artist_dict[artist2][random.randint(0, len(artist_dict[artist2])-1)] # random painting by that artist

                    touple_array.append([painting1, painting2])
                    answer.append(1 if artist1 == artist2 else 0)

                elif sum(answer) <= 0.5*len(answer): # If more negatives - pick a random positive

                    artist1 = artist
                    painting1 = painting

                    artist2 = artist1 # same artist

                    painting2 = painting1
                    while painting2 == painting1:
                        painting2 = artist_dict[artist2][random.randint(0, len(artist_dict[artist2])-1)] # different painting
                                                         
                    tries = 0
                    while ([painting1, painting2] in touple_array) or ([painting2, painting1] in touple_array): # if we already have that touple - try again

                        if tries == 100: # If tried a 100 times - give up
                            break
                                                         
                        painting2 = painting1
                        while painting2 == painting1:
                            painting2 = artist_dict[artist2][random.randint(0, len(artist_dict[artist2])-1)] # different painting

                        tries += 1
                                                             
                    if tries == 100: # if failed - don't make more touples for this painting
                        break

                    touple_array.append([painting1, painting2])
                    answer.append(1 if artist1 == artist2 else 0)
                


    # Shuffle the arrays so we do not to get [1,0,1,0,1,0 ... ]
    list_to_shuffle = list(zip(touple_array, answer))
    random.shuffle(list_to_shuffle)
    touple_array, answer = zip(*list_to_shuffle)
    
    return touple_array, answer
            
