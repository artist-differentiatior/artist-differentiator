# artist-differentiator
A CNN trained with triplets to identify if two paintings are painted by same artist
# Requirements
The requirements for this project can be found in: requirements.txt.
# Running
To initialize the project you need to set up a directory containing the triplets. To do so you can use the script: `preprocess_images.py`. 
This script requires a directory contaning training images and a csv-file with information about the images in the source folder. The csv-file should be name `train_info.csv` and it should have the information:`filename`, `artist` and `style` of the images. If the directory containing the images is named `sample_images` you should run:

`python preprocess.py sample_triplet/`

To start training the model:

`python train_nn.py`

generate data

`python preprocess_data.py --source sample_data/ --info-file data_info.csv --test-ratio 1`

test model

`python evaluate.py --test-path test_data/ --weight-path weights_3_artists.mat --style-layers-indices 3 4 --type test`

generate pca

`python apply_pca.py --weights weights_3_artists.mat --csv data_info.csv --source test_data/ --type test --style-layers-indices 3 4 --mode pca`

