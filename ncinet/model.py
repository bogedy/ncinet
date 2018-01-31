"""
Implements the core neural networks.
"""

# code adapted from tutorials:
# https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from .layers import conv_layer, fc_layer, NciKeys
from .config_hyper import EncoderConfig, InfConfig


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


def ae_encoder(prints, config, training=True):
    # type: (tf.Tensor, EncoderConfig, bool) -> tf.Tensor
    """Encoder used in the autoencoder and inference networks."""
    collection = NciKeys.AE_ENCODER_VARIABLES
    pool = prints

    for k in range(config.n_layers):
        name = "conv{}".format(k)
        k_size = (config.filter_size[k],) * 2
        conv = conv_layer(inputs=pool, filters=config.n_filters[k], kernel_size=k_size, trainable=training,
                          collection=collection, batch_norm=(k == 0), wd=config.reg_weight[k], name=name)
        _activation_summary(conv)
        pool = tf.layers.max_pooling2d(conv, pool_size=(2, 2), strides=(2, 2), padding='same')

    return pool


def ae_decoder(encoded, config):
    # type: (tf.Tensor, EncoderConfig) -> tf.Tensor
    """Decoder used in the autoencoder."""
    collection = NciKeys.AE_DECODER_VARIABLES
    conv = encoded

    for k in range(config.n_layers):
        i = -(k+1)
        name = "convT{}".format(k)
        k_size = (config.filter_size[i],)*2
        i_size = (config.init_dim[i],)*2

        upsample = tf.image.resize_images(conv, size=i_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv = conv_layer(inputs=upsample, filters=config.n_filters[i], kernel_size=k_size, wd=config.reg_weight[i],
                          collection=collection, name=name)

    # 100x100x1
    logits = conv_layer(inputs=conv, filters=1, kernel_size=(5, 5), activation=None,
                        collection=collection, name="ConvT4")

    return logits


def autoencoder(prints, config, training=True):
    """Constructs the autoencoder network."""
    tf.summary.image("fingerprints", prints)
    encoded = ae_encoder(prints, config, training=training)
    predict = ae_decoder(encoded, config)
    tf.summary.image("predicts", tf.nn.sigmoid(predict))
    return predict


def inf_classify(prints, config, training=True):
    # type: (tf.Tensor, InfConfig, bool) -> tf.Tensor
    """Constructs a classifier inference network"""
    tf.summary.image("fingerprints", prints)
    encoded = ae_encoder(prints, config.encoder_config, training=False)
    _activation_summary(encoded)

    fc = encoded
    for k in range(config.n_hidden):
        name = "fc{}".format(k)
        fc = fc_layer(fc, config.dim_hidden[k], name=name, collection=NciKeys.INF_VARIABLES, trainable=training)
        _activation_summary(fc)

    logits = fc_layer(fc, config.n_logits, name="logits", activation=None,
                      collection=NciKeys.INF_VARIABLES, trainable=training)
    _activation_summary(logits)

    return logits
