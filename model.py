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


def inference(seqs):
    # creates the CNN

    # initial batch normalization
    mean1, variance1 = tf.nn.moments(seqs, [0], keep_dims=True)
    norm1 = tf.nn.batch_normalization(seqs, mean1, variance1, 0.001)

    # first layer
    with tf.variable_scope("conv1") as scope:
        kernel = tf.Variable("kernel_weights1",
                             shape=[KER_LENGTH, 1, BATCH_SIZE],
                             stddev=5e-2,
                             wd=0.0)
        conv = tf.nn.conv1d(norm1, kernel, stride=1, padding="SAME")
        biases = _variable_on_cpu("biases",
                                  [BATCH_SIZE],
                                  tf.truncated_normal_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # no pooling layer because of the nature of stock data precision

    # normalization layer with batch normalization hidden -> hidden
    mean2, variance2 = tf.nn.moments(conv1, [0], keep_dims=True)
    norm2 = tf.nn.batch_normalization(conv1, mean2, variance2, 0.001)

    # second layer
    with tf.variable_scope("conv2") as scope:
        kernel = tf.Variable("kernel_weights2",
                             shape=[KER_LENGTH, 1, BATCH_SIZE],
                             stddev=5e-2,
                             wd=0.0)
        conv = tf.nn.conv1d(norm2, kernel, stride=1, padding="SAME")
        biases = _variable_on_cpu("biases",
                                  [BATCH_SIZE],
                                  tf.truncated_normal_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # dense layer
    with tf.variable_scope("dense") as scope:
