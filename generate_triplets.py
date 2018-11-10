
import csv
import random
import os


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
        style_dict = {} # key=style, value=[[file_name, artist], ...]
        info = info_reader.next()
        
        artist_index = info.index("artist")
        filename_index = info.index("filename")
        style_index = info.index("style")

        for row in info_reader:
            
            artist_name = row[artist_index]
            style_name = row[style_index]
            file_name = row[filename_index]

            if file_name not in file_names:
                continue
            
            artist_dict_info = [file_name, style_name]
            style_dict_info = [file_name, artist_name]
            
            if artist_name not in artist_dict:
                artist_dict[artist_name] = [artist_dict_info]
            else:
                artist_dict[artist_name].append(artist_dict_info)

            if style_name not in style_dict:
                style_dict[style_name] = [style_dict_info]
            else:
                style_dict[style_name].append(style_dict_info)

        
    return artist_dict, style_dict

def generate_triplets(csv_file_path, paintings_file_path):

    '''
    Generates triplets from .cvs file in csv_file_path.

    Args:
        csv_file_path: (str) path to csv file containing info about dataset
        paintings_file_path: (str) path to directory containing dataset
    '''

    artist_dict, style_dict = _parse_info_file(csv_file_path, paintings_file_path)
    triplet_array = []

    
    for artist, paintings_by_artist in artist_dict.iteritems(): # Iterate over all artists
        
        assert len(paintings_by_artist) > 1, "Just 1 painting from %r, can't generate triplet!" %artist
        
        for i in range(0,len(paintings_by_artist) - 1): # Iterate over paintings by current artist
            anchor_file_name = paintings_by_artist[i][0] # Pick current painting as anchor
            positive_file_name = paintings_by_artist[i+1][0] # Pick next painting by same artist as positive

            anchor_style = paintings_by_artist[i][1]
            paintings_with_style = style_dict[anchor_style] # retrieve all paintings with same style as anchor

            negative_file_name = ''
            painting_indices = list(range(len(paintings_with_style)))
            random.shuffle(painting_indices)
            for j in painting_indices:
                negative_artist = paintings_with_style[j][1]
                if artist != negative_artist:
                    negative_file_name = paintings_with_style[j][0] # Pick next painting with different artist but same style as negative

            if negative_file_name == '': # If no such painting exists
                while True: 
                    random_artist = random.choice(list(artist_dict)) # Pick a random other artist
                    if random_artist != artist:
                        break
                paintings_by_random_artist = artist_dict[random_artist]
                random_index = random.randrange(0, len(paintings_by_random_artist)-1) # Pick a random painting by that artist
                negative_file_name = paintings_by_random_artist[random_index][0]
            
            triplet_array.append([anchor_file_name, positive_file_name, negative_file_name]) # Add the new triplet to array



    return triplet_array

    
    
                

def main():

    print(generate_triplets("new_train_info.csv", "sample_triplet/"))
    

