import csv
import os
import random
import copy

import numpy as np


def parse_info_file(csv_file_path, paintings_file_path):

    '''
    Parses info file. Creates dictionary with keyword as artists and the corresponding value as an 
    array of paintings.

    Args:
        csv_file_path: (str) path to csv file containing info about dataset
        paintings_file_path: (str) path to directory containing dataset
    '''
    
    file_names = os.listdir(paintings_file_path)

    with open(csv_file_path, "r") as info_file:

        info_reader = csv.reader(info_file, dialect="excel", delimiter=",", quotechar="\"")

        # key=artist, value=[[file_name, style], ...]
        artist_dict = {} 
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

    n_artists_before = len(artist_dict.keys())
    
    artists_to_delete = []

    for artist, paintings in artist_dict.iteritems():
        if len(paintings) <= 1:
            print('Deleted artist: {}'.format(artist))
            artists_to_delete.append(artist)

    for artist in artists_to_delete:
	    del artist_dict[artist]

    n_artists_after = len(artist_dict.keys())
    print('Artists before: {}\nArtists after: {}'.format(n_artists_before, n_artists_after))  
      
    return artist_dict


def parse_info_file_triplets(csv_file_path, paintings_file_path):

    '''
    Parses info file for paintings in triplet file path. Creates dictionary with keyword as artists and the corresponding value as an 
    array of paintings.

    Args:
        csv_file_path: (str) path to csv file containing info about dataset
        paintings_file_path: (str) path to directory containing triplets
    '''
    file_names = []
    
    file_names = os.listdir(paintings_file_path)

    for i in range(len(file_names)):
        file_names[i] = file_names[i].split('-')[1]

    with open(csv_file_path, "r") as info_file:

        info_reader = csv.reader(info_file, dialect="excel", delimiter=",", quotechar="\"")

        # key=artist, value=[[file_name, style], ...]
        artist_dict = {}
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


def convert_dict_to_list(artist_dict):

    """
    Converts a dictionary containing artists and painting into a list with touples where 
    each touples consist of painting names and the corresponding artist.

    Args:
    artist_dict: (dict) Dictionary, keys: artist, value list with paintings
    """

    new_list = []
    for artist, paintings in artist_dict.iteritems():
        for painting in paintings:
            new_list.append([artist, painting])

    return np.array(new_list)


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

                # Add the new triplet to array
                triplet_array.append([painting, positive_painting, negative_painting])

    return triplet_array


def generate_touple(artist_dict, num):

    '''
    Generates touple from a dictonary with artists as keys with painted paintings as value

    Args:
    artist_dict: (dict) Key (string): artist, value (array) corresponding paintings
    num: (int) Number of times to generate test-pairs from data
    '''

    nr_paintings = 0
    for artist in artist_dict.keys():
        nr_paintings += len(artist_dict[artist])

    if nr_paintings % 2 == 1:
        rand_artist = random.choice(artist_dict.keys())
        rand_painting =  random.sample(artist_dict[rand_artist], 1)[0]
        artist_dict[rand_artist].remove(rand_painting)

    nr_paintings = 0
    for artist in artist_dict.keys():
        nr_paintings += len(artist_dict[artist])

    assert nr_paintings % 2 == 0

    touple_array = []
    answer = []

    num_same = 0
    num_diff = 0

    original_artist_dict = copy.deepcopy(artist_dict)

    for count in range(num):

        artist_dict = copy.deepcopy(original_artist_dict)

        while artist_dict:

            rand_artist = random.choice(artist_dict.keys())

            if (num_same <= num_diff or len(artist_dict.keys()) == 1) and len(artist_dict[rand_artist]) > 1:
            
                two_paintings = random.sample(artist_dict[rand_artist], 2)

                painting1 = two_paintings[0]
                painting2 = two_paintings[1]

                artist_dict[rand_artist].remove(painting1)
                artist_dict[rand_artist].remove(painting2)

                if len(artist_dict[rand_artist]) == 0:
                    del artist_dict[rand_artist]
            
                touple_array.append([painting1, painting2])
                answer.append(1)

                num_same += 1

            else:

                other_artists = artist_dict.keys()
                other_artists.remove(rand_artist)

                second_artist = random.choice(other_artists)

                painting1 = random.sample(artist_dict[rand_artist], 1)[0]

                artist_dict[rand_artist].remove(painting1)

                if len(artist_dict[rand_artist]) == 0:
                    del artist_dict[rand_artist]

                painting2 = random.sample(artist_dict[second_artist], 1)[0]

                artist_dict[second_artist].remove(painting2)

                if len(artist_dict[second_artist]) == 0:
                    del artist_dict[second_artist]

                touple_array.append([painting1, painting2])
                answer.append(0)

                num_diff += 1

    list_to_shuffle = list(zip(touple_array, answer))
    random.shuffle(list_to_shuffle)
    touple_array, answer = zip(*list_to_shuffle)

    return touple_array, answer
