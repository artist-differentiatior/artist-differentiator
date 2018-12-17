# artist-differentiator
A siamese CNN model trained with triplet loss to identify if two paintings are painted by the same artist.
We have used the dataset provided in the 2016 Kaggle competition ["Painter by Numbers"](https://www.kaggle.com/c/painter-by-numbers).

## How it works
The approach used is outlined in the figure below.

<p align="center"> 
<img src=https://github.com/josefmal/artist-differentiator/blob/master/figures/siamese_net.svg>
</p>

From a training set of paintings we form triplets, each consisting of a base painting (anchor), another painting by the same artist (positive) and a third painting by a different artist (negative). When training, the paintings of a triplet is fed through a set of pretrained siamese CNNs and an encoding of the style of each painting is extracted (much like in neural
style transfer). We then train the network further, on the triplet loss formed in terms of the style encodings of the anchor, positive and negative paintings. To evaluate if two paintings were made by the same artist, the paintings are fed through the same, trained network and the distance between their style encodings are compared.

For detailed descriptions on the architectures and methods used, refer to our paper on this project, available [here](https://drive.google.com/drive/folders/1oRqil4zFwI-TcJiYQLB2p4_2Po44p0le?usp=sharing).

## Results
Some PCA/t-SNE plots of the style encodings learned by the model are presented below. Each point represents the style encoding of a painting and each color represents a different artist. The approach works well for a limited number of artists, but as the number of artists increases it becomes harder for the model to distinguish the styles.

<p align="center"> 
<img src=https://github.com/josefmal/artist-differentiator/blob/master/figures/results.svg >
</p>

Some possible improvements for this approach include exploring alternative methods for evaluating on a pair of images, given their style encodings. Another possible improvement would be to find an way to algorithmically generate non-trivial triplets for training, as the triplets currently used for training are completely random. See our [paper](https://drive.google.com/drive/folders/1oRqil4zFwI-TcJiYQLB2p4_2Po44p0le?usp=sharing) for details.


## Requirements
The code is written for Python 2.7 in TensorFlow. The required modules for this project can be found in `requirements.txt`.

## Running
Here we outline the different scripts and how to run them.

### Preprocessing
First you need to preprocess your data with the script `preprocess_data.py`. This will rescale the images and form triplets or pairs of images, depending on the configured options. To preprocess the provided sample data (purely as test data) run the command: 

`python preprocess_data.py --source sample_data/ --info-file data_info.csv --test-ratio 1`

If you want to create triplets for training, the script has settings for doing so. To see options, run the command

`python preprocess_data.py -h`

### Training
To train a model you can use the script `train_parser.py`. To list the available settings run the command:

`python train_parser.py -h`

### Evaluating
To evaluate on a preprocessed set of images (test or train), use the script `evaluate.py`. For example, to evaluate on the provided sample images (after preprocessing), using pretrained weights `3_artists_weights.mat`, available [here](https://drive.google.com/drive/folders/1oRqil4zFwI-TcJiYQLB2p4_2Po44p0le?usp=sharing), run the following command:

`python evaluate.py --test-path test_data/ --weight-path weights_3_artists.mat --style-layers-indices 3 4 --type test`

To list the available settings, run the command:

`python evaluate.py -h`

### Plotting PCA/t-SNE
To generate PCA/t-SNE plots of the learned style encodings, use the script `apply_pca.py`. For examlpe, to generate a PCA plot of the style encodings for the sample images (with the weights linked above), run the following command:

`python apply_pca.py --weights weights_3_artists.mat --csv data_info.csv --source test_data/ --type test --style-layers-indices 3 4 --mode pca`

To list the available settings, run the command:

`python apply_pca.py -h`

