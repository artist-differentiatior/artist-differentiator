
import csv


def _parse_info_file(file_path):

    with open(file_path, "r") as info_file:

        info_reader = csv.reader(info_file, dialect="excel", delimiter=",", quotechar="\"")

        info_array = []
        info_reader.next()

        for row in info_reader:
            info_array.append(row)
        
    return info_array

def generate_triplets():

    info_array = _parse_info_file("./data_info/train_info.csv")

    anchor_images = []
    anchor_artists = []
    anchor_styles = []
    
    positive_images = []
    
    negative_images = []

    # Choose anchors - Currently first image of each artist
    for row in info_array:
        image = row[0]
        artist = row[1]
        style = row[3]

        if artist in anchor_artists:
            continue

        anchor_images.append(image)
        anchor_artists.append(artist)
        anchor_styles.append(style)

    # Choose positives - Currently next occurrance of image by same artist as anchor
    i = 0
    for row in info_array:

        if i >= len(anchor_images):
            break
        
        image = row[0]
        artist = row[1]
        style = row[3]

        if image in anchor_images:
            continue
        if artist != anchor_artists[i]:
            continue

        positive_images.append(image)
        i += 1

    

    # Choose negatives - Currently next occurrance of an image NOT by the same artist, but with the same style as anchor
    i = 0
    for row in info_array:

        if i >= len(anchor_images):
            break

        image = row[0]
        artist = row[1]
        style = row[3]

        if artist == anchor_artists[i]:
            continue
        if style != anchor_styles[i]:
            continue

        negative_images.append(image)
        i += 1

    

    print(len(anchor_images))
    print(len(info_array))
    
                

def main():

    generate_triplets()
    

if __name__ == "__main__":
    main()
