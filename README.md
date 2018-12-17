# artist-differentiator
A siamese CNN model trained with triplet loss to identify if two paintings are painted by the same artist.
This is one attempt at solving the problem presented in the 2016 Kaggle competition ["Painter by Numbers"](https://www.kaggle.com/c/painter-by-numbers).

## How it works
The approach used is outlined in the figure below.

<p align="center"> 
<img src="https://github.com/josefmal/artist-differentiator/blob/master/figures/siamese_net.svg ">
</p>


From a training set of paintings we form triplets, each consisting of a base painting (anchor), another painting by the same artist (positive) and a third painting by a different artist (negative). When training, the paintings of a triplet is fed through a set of pretrained siamese CNNs and an encoding of the style of each painting is extracted (much like in neural
style transfer). We then train the network further, on the triplet loss formed in terms of the style encodings of the anchor, positive and negative paintings. To evaluate if two paintings were made by the same artist, the paintings are fed through the same, trained network and the distance between their style encodings are compared.

For detailed descriptions on the architectures and methods used, refer to our paper on this project, available [here](https://drive.google.com/drive/folders/1oRqil4zFwI-TcJiYQLB2p4_2Po44p0le?usp=sharing).

## Results
Some PCA/t-SNE plots of the style encodings learned by the model are presented below. Each point represents the style encoding of a painting and each color represents a different artist. The approach works well for a limited number of artists, but as the number of artists increases it becomes harder for the model to distinguish the styles.

![](https://github.com/josefmal/artist-differentiator/blob/master/figures/results.svg "PCA/t-SNE results")

Some possible improvements for this approach include exploring alternative methods for evaluating on a pair of images, given their style encodings. Another possible improvement would be to find an way to algorithmically generate non-trivial triplets for training, as the triplets currently used for training are completely random. See our [paper](https://drive.google.com/drive/folders/1oRqil4zFwI-TcJiYQLB2p4_2Po44p0le?usp=sharing) for details.


## Requirements
The requirements for this project can be found in: requirements.txt.
## Running
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

