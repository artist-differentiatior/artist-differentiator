import os

import numpy as np
import scipy.misc

from train_nn import train_nn

import math
from argparse import ArgumentParser

from PIL import Image

# default arguments
LEARNING_RATE = 1e-4
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
EPOCHS = 1
VGG_PATH = 'vgg_net_original.mat'
LOSS_THRESHOLD = 1e12
POSITIVE_WEIGHT = 1

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
    parser.add_argument('--positive-weight', type=float,
            dest='positive_weight', help='Loss: weight of AP-distance (default %(default)s)',
            metavar='POSITIVE_WEIGHT', default=POSITIVE_WEIGHT)
    parser.add_argument('--mini-batch-size', type=int,
            dest='mini_batch_size', help='mini batch size (default %(default)s)',
            metavar='MINI_BATCH_SIZE', default=4)
    parser.add_argument('--device',
            dest='device', help='device - cpu or gpu (default %(default)s)',
            metavar='DEVICE', default='cpu')
    parser.add_argument('--loss-threshold', type=int,
            dest='loss_threshold', help='mini batch size (default %(default)s)',
            metavar='LOSS_THRESHOLD', default=LOSS_THRESHOLD)
    parser.add_argument('--style-layers-indices', nargs='+', type=int,
            dest='style_layers_indices', help='Which of the 5 (index 0-4) style layers to use (default %(default)s)',
                        metavar='STYLE_LAYERS_INDICES', default=[3,4])
    
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

    print(" ------------ Current Options ------------")
    print("Network: %s" % options.network)
    print("Epochs: %d" % options.epochs)
    print("Learning rate: %e" % options.learning_rate)
    print("Mini batch size: %d" % options.mini_batch_size)
    print("Weight destination: %s" % options.name)
    print("Loss threshold: %e" % options.loss_threshold)
    print("Style layers: %s" % options.style_layers_indices)
    print("Using checkpoints: %s" % options.checkpoints)
    print("Device: %s" % options.device)
    print("------------------------------------------")
    print("")
    
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
        loss_threshold=options.loss_threshold,
        positive_weight=options.positive_weight,
        style_layers_indices=options.style_layers_indices
    )

if __name__ == '__main__':
    main()
