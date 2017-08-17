# many helper functions taken from TensorFlow CIFAR tutorial

import os
import re
import sys
import tarfile

import tensorflow as tf

import parse

SEQ_LENGTH = parse.SEQ_LENGTH
KER_LENGTH = 7
NUM_CLASSES = parse.NUM_CLASSES
BATCH_SIZE = parse.BATCH_SIZE
# NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
# NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def create_network(seqs):
    # creates the NN
    # Structure:
    with tf.variable_scope("conv1") as scope:
        kernel = tf.Variable("kernel_weights",
                             shape=[KER_LENGTH, 1, BATCH_SIZE],
                             stddev=1e-1,
                             wd=0.0)
        conv = tf.nn.conv1d(seqs, kernel, stride=1, padding="SAME")
        biases = _variable_on_cpu("biases", [BATCH_SIZE], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # no pooling layer

    norm1 = tf.nn.lrn()