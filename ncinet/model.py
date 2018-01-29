
# code adapted from tutorials:
# https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .layers import conv_layer, fc_layer, NciKeys


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
    n_layers = 3
    n_filter_par = [32, 32, 16]
    k_size_par = [5, 5, 5]
    wd_par = [0.001, 0.001, 0.001]

    pool = prints

    for k in range(n_layers):
        name = "conv{}".format(k)
        k_size = (k_size_par[k], k_size_par[k])
        conv = conv_layer(inputs=pool, filters=n_filter_par[k], kernel_size=k_size, trainable=training,
                          collection=collection, batch_norm=(k == 0), wd=wd_par[k], name=name)
        _activation_summary(conv)
        pool = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), padding='same')

    return pool


def top_convnet_autoenc(encoded):
    """Decoder used in the autoencoder."""
    collection = NciKeys.AE_DECODER_VARIABLES
    n_layers = 3
    n_filter_par = [16, 32, 32]
    k_size_par = [5, 5, 5]
    wd_par = [None, None, None]
    i_start = 25

    conv = encoded

    for k in range(n_layers):
        name = "convT{}".format(k)
        k_size = (k_size_par[k], k_size_par[k])
        i_size = (i_start*(k + 1), i_start*(k + 1))

        upsample = tf.image.resize_images(conv, size=i_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv = conv_layer(inputs=upsample, filters=n_filter_par[k], kernel_size=k_size, wd=wd_par[k],
                          collection=collection, name=name)

    # 100x100x1
    logits = conv_layer(inputs=conv, filters=1, kernel_size=(5, 5), activation=None,
                        collection=collection, name="ConvT4")

    return logits


def autoencoder(prints, training=True):
    """Constructs the autoencoder network."""
    tf.summary.image("fingerprints", prints)
    encoded = bottom_convnet(prints, training=training)
    predict = top_convnet_autoenc(encoded)
    tf.summary.image("predicts", tf.nn.sigmoid(predict))
    return predict


def topo_classify(prints, training=True):
    """Constructs the inference network"""
    tf.summary.image("fingerprints", prints)
    encoded = bottom_convnet(prints, training=False)

    n_hidden = 1
    dim_hidden = [256]

    fc = encoded
    for k in range(n_hidden):
        name = "fc{}".format(k)
        fc = fc_layer(fc, dim_hidden[k], name=name, collection=NciKeys.INF_VARIABLES, trainable=training)
        _activation_summary(fc)

    logits = fc_layer(fc, 4, name="fc{}".format(n_hidden), activation=None,
                      collection=NciKeys.INF_VARIABLES, trainable=training)
    _activation_summary(logits)

    return logits


def sign_classify(prints, training=True):
    """Construct a network to classify stability score > 0"""
    tf.summary.image("fingerprints", prints)
    encoded = bottom_convnet(prints, training=False)
    _activation_summary(encoded)

    n_hidden = 1
    dim_hidden = [128]

    fc = encoded
    for k in range(n_hidden):
        name = "fc{}".format(k)
        fc = fc_layer(fc, dim_hidden[k], name=name, collection=NciKeys.INF_VARIABLES, trainable=training)
        _activation_summary(fc)

    logits = fc_layer(fc, 2, name="fc{}".format(n_hidden), activation=None,
                      collection=NciKeys.INF_VARIABLES, trainable=training)
    _activation_summary(logits)

    return logits
