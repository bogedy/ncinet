
# code adapted from tutorials:
# https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from layers import conv_layer, fc_layer, NciKeys

# TODO: consider binding some actions to CPU

# Directory where summaries and checkpoints are written.
#WORK_DIR = '/work/05187/ams13/maverick/Working/TensorFlow/ncinet'
WORK_DIR = 'C:/Users/schan/Documents/TF_run'


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    """

    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                      tf.nn.zero_fraction(x))


def bottom_convnet(prints, training=True):
    """Encoder used in the autoencoder and inference networks."""
    collection = NciKeys.AE_ENCODER_VARIABLES
    # 100x100x32
    conv1 = conv_layer(inputs=prints, filters=32, kernel_size=(5, 5), training=training,
                       collection=collection, batch_norm=True, wd=0.001, name="conv1")
    _activation_summary(conv1)
    # 50x50x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')

    # 50x50x32
    conv2 = conv_layer(inputs=maxpool1, filters=32, kernel_size=(5, 5), training=training,
                       collection=collection, wd=0.001, name="conv2")
    _activation_summary(conv2)
    # 25x25x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')

    # 25x25x16
    conv3 = conv_layer(inputs=maxpool2, filters=16, kernel_size=(5, 5), training=training,
                       collection=collection, wd=0.001, name="conv3")
    _activation_summary(conv3)
    # 13x13x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same', name="encoded")

    return encoded


def top_convnet_autoenc(encoded):
    """Decoder used in the autoencoder."""
    collection = NciKeys.AE_DECODER_VARIABLES
    # 25x25x16
    upsample1 = tf.image.resize_images(encoded, size=(25, 25), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # 25x25x16
    conv4 = conv_layer(inputs=upsample1, filters=16, kernel_size=(5, 5), collection=collection, name="convT1")
    # 50x50x16
    upsample2 = tf.image.resize_images(conv4, size=(50, 50), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # 50x50x32
    conv5 = conv_layer(inputs=upsample2, filters=32, kernel_size=(5, 5), collection=collection, name="convT2")
    # 100x100x32
    upsample3 = tf.image.resize_images(conv5, size=(100, 100), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # 100x100x32
    conv6 = conv_layer(inputs=upsample3, filters=32, kernel_size=(5, 5), collection=collection, name="ConvT3")

    # 100x100x1
    logits = conv_layer(inputs=conv6, filters=1, kernel_size=(5, 5), activation=None,
                        collection=collection, name="ConvT4")

    return logits


def autoencoder(prints, training=True):
    """Constructs the autoencoder network."""
    tf.summary.image("fingerprints", prints)
    base = bottom_convnet(prints, training=training)
    predict = top_convnet_autoenc(base)
    tf.summary.image("predicts", tf.nn.sigmoid(predict))
    return predict


def inference(prints, training=True):
    """Constructs the inference network"""
    tf.summary.image("fingerprints", prints)
    base = bottom_convnet(prints, training=False)
    # TODO: dont hardcode this shape
    flat = tf.reshape(base, [-1, 13*13*16])
    _activation_summary(flat)
    fc1 = fc_layer(flat, 512, name="fc1", collection=NciKeys.INF_VARIABLES, training=training)
    _activation_summary(fc1)
    logits = fc_layer(fc1, 4, name="fc2", activation=None, collection=NciKeys.INF_VARIABLES, training=training)
    _activation_summary(logits)
    return logits


def sign_classify(prints, training=True):
    """Construct a network to classify stability score > 0"""
    tf.summary.image("fingerprints", prints)
    base = bottom_convnet(prints, training=False)
    _activation_summary(base)
    fc1 = fc_layer(base, 128, name="fc1", collection=NciKeys.INF_VARIABLES, training=training)
    _activation_summary(fc1)
    logits = fc_layer(fc1, 2, name="fc2", activation=None, collection=NciKeys.INF_VARIABLES, training=training)
    _activation_summary(logits)
    return logits
