
import csv
import random
import os
import numpy as np


def _parse_info_file(csv_file_path, paintings_file_path):

    '''
    Parses info file. Creates dictionary with keyword as artists and the corresponding value an 
    array of paintings, and dictionary with keyword as style and corresponding value an array of paintings.

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

def generate_triplets(csv_file_path, paintings_file_path, num_anchors=10):

    '''
    Generates triplets from .cvs file in csv_file_path.

    Args:
        csv_file_path: (str) path to csv file containing info about dataset
        paintings_file_path: (str) path to directory containing dataset
    '''

    artist_dict = _parse_info_file(csv_file_path, paintings_file_path)
    triplet_array = []

    assert len(artist_dict) > 1, "Just 1 artist"

    for artist, paintings_by_artist in artist_dict.iteritems(): # Iterate over all artists
        
        assert len(paintings_by_artist) > 1, "Just 1 painting from %r, can't generate triplet!" %artist
        
        for index, painting in enumerate(paintings_by_artist): # Iterate over paintings by current artist

            anchor_file_name = painting # Pick current painting as anchor
            
            while True:
                indices = np.random.randint(0,len(paintings_by_artist), 10)
                if index not in indices:
                    break

            for i in indices:
                positive_file_name = paintings_by_artist[i]

                artists = list(artist_dict.keys())
                artists.remove(artist) #remove current artist

                random_artist = random.choice(artists)
                random_painting = random.choice(artist_dict[random_artist])
                negative_file_name = random_painting

                triplet_array.append([anchor_file_name, positive_file_name, negative_file_name]) # Add the new triplet to array


    return triplet_array

    
    
                

def main():

    print(generate_triplets("new_train_info.csv", "test2/"))

if __name__ == "__main__":
    main()
    

