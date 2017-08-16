# many helper functions taken from TensorFlow CIFAR tutorial

import os
import re
import sys
import tarfile

import tensorflow as tf

import parse

SEQ_LENGTH = parse.SEQ_LENGTH
NUM_CLASSES = parse.NUM_CLASSES
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def create_network():
    # creates the NN
    # Structure:
    pass
