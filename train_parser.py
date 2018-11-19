import os

import numpy as np
import scipy.misc

from train_nn import train_nn

import math
from argparse import ArgumentParser

from PIL import Image

# default arguments
CONTENT_WEIGHT = 5e0
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 5e2
TV_WEIGHT = 1e2
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
EPOCHS = 1
VGG_PATH = 'vgg_net_original.mat'
POOLING = 'max'
LOSS_THRESHOLD = 1e12

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int,
            dest='epochs', help='epochs (default %(default)s)',
                        metavar='EPOCHS', default=EPOCHS)
    parser.add_argument('--network',
            dest='network', help='path to network parameters (default %(default)s)',
            metavar='VGG_PATH', default=VGG_PATH)
    parser.add_argument('--dest',
            dest='name', help='name/path of the trained weights',
            metavar='DEST', required=True)
    parser.add_argument('--checkpoints', type=bool,
            dest='checkpoints', help='If you want to resume from a saved checkpoint (default %(default)s)',
            metavar='CHECKPOINTS', default=False)
    parser.add_argument('--learning-rate', type=float,
            dest='learning_rate', help='learning rate (default %(default)s)',
            metavar='LEARNING_RATE', default=LEARNING_RATE)
    parser.add_argument('--beta1', type=float,
            dest='beta1', help='Adam: beta1 parameter (default %(default)s)',
            metavar='BETA1', default=BETA1)
    parser.add_argument('--beta2', type=float,
            dest='beta2', help='Adam: beta2 parameter (default %(default)s)',
            metavar='BETA2', default=BETA2)
    parser.add_argument('--eps', type=float,
            dest='epsilon', help='Adam: epsilon parameter (default %(default)s)',
            metavar='EPSILON', default=EPSILON)
    parser.add_argument('--mini-batch-size', type=int,
            dest='mini_batch_size', help='mini batch size (default %(default)s)'
            metavar='MINI_BATCH_SIZE', default=4)
    parser.add_argument('--device',
            dest='device', help='device - cpu or gpu (default %(default)s)',
            metavar='DEVICE', default='cpu')
    parser.add_argument('--loss-threshold', 
            dest='loss_threshold', help='mini batch size (default %(default)s)',
                        metavar='LOSS_THRESHOLD', default=LOSS_THRESHOLD)
    
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()

    if not os.path.isfile(options.network):
        parser.error("Network %s does not exist. (Did you forget to download it?)" % options.network)

    if os.path.isfile(options.name + ".mat"):
        print("Warning! " + options.name + ".mat" + " does already exist, do you want to overwrite it? (Y/N)")
        answer = raw_input()
        if answer != "Y" and answer != "y":
                print("Exiting, rerun with another filename.")
                exit()
                
    if options.device == 'cpu':
        options.device = '/cpu:0'
    elif options.device == 'gpu':
        options.device = '/gpu:0'
    else:
        parser.error('Invalid device. Choose cpu or gpu')
        exit()
    

    train_nn(
        network=options.network,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        beta1=options.beta1,
        beta2=options.beta2,
        epsilon=options.epsilon,
        batch_size=options.mini_batch_size,
        save_file_name=options.name,
        checkpoints=options.checkpoints,
        device_name=options.device,
        loss_threshold=options.loss_threshold
    )

if __name__ == '__main__':
    main()
