
import csv


def _parse_info_file(file_path):

    '''
    Parse info file. Create dictionary with keyword as artists and the correspnding value an 
    array with paintings of that artist and info about those paintings.
    '''

    with open(file_path, "r") as info_file:

        info_reader = csv.reader(info_file, dialect="excel", delimiter=",", quotechar="\"")

        info_dict = {}
        info = info_reader.next()

        artist_index = info.index("artist")
        filename_index = info.index("filename")
        style_index = info.index("style")

        for row in info_reader:
            artist_name = row[artist_index]
            painting_info = [row[filename_index], row[style_index]]
            if artist_name not in info_dict:
                info_dict[artist_name] = [painting_info]
            else:
                info_dict[artist_name].append(painting_info)

        
    return info_dict

def generate_triplets(file_path):

    info_dict = _parse_info_file(file_path)
    tripplet_array = []

    for artist, painting_info in info_dict.iteritems():
        assert len(painting_info) > 1, "Just 1 painting from %r, can't generate tripplet!" %artist
        for i in range(0,len(painting_info) - 1):
            tripplet_array.append([painting_info[i][0], painting_info[i + 1][0]])



    return tripplet_array

    
    
                

def main():

    print(generate_triplets("test.csv"))
    

if __name__ == "__main__":
    main()
