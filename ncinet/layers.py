"""
Wrappers for network layers incorporating L2 regularization
"""

from __future__ import division

import numpy as np
import tensorflow as tf


class NciKeys:
    """Keys for grouping different sections of the network"""
    AE_ENCODER_VARIABLES = "_NciKeys_ae_encoder_var"
    AE_DECODER_VARIABLES = "_NciKeys_ae_decoder_var"
    INF_VARIABLES = "_NciKeys_inf_var"


def weight_var_init(n):
    """Initialize weights from normal distribution

    Follows recommendation of Glorot et al. (2006) to draw from a distribution
    with `var = 2.0/n` where `n` is the fan-in of the layer.
    """
    n = tf.cast(n, tf.float32)
    std_dev = tf.sqrt(2.0 / n)
    return tf.truncated_normal_initializer(mean=0.0, stddev=std_dev,
                                           dtype=tf.float32)


def _variable_with_weight_decay(name, shape, initializer, weight_decay, trainable=True):
    """Helper to create an initialized Variable with L2 regularization.

    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: tensorflow initializer for the variable
        weight_decay: add L2Loss weight decay multiplied by this float.
            If None, weight decay is not added for this Variable.
        trainable: bool, whether variable is trainable
    Returns:
        Variable Tensor
    """
    var = tf.get_variable(name, shape=shape,
                          initializer=initializer,
                          dtype=tf.float32,
                          trainable=trainable)

    if trainable and weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var),
                                   weight_decay, name='weight_loss')

        tf.add_to_collection('losses', weight_decay)

    return var


def conv_layer(inputs, filters, kernel_size, padding='SAME',
               activation=tf.nn.relu, trainable=True, collection=None,
               batch_norm=False, wd=None, name='conv'):
    """Wraps a conv2D layer with biases, normalization, and regularization"""

    with tf.variable_scope(name):
        channels = inputs.get_shape().as_list()[-1]
        shape = [kernel_size[0], kernel_size[1], channels, filters]
        ker = _variable_with_weight_decay(name="Kernel", shape=shape,
                                          initializer=weight_var_init(np.prod(shape[:-1])),
                                          weight_decay=wd, trainable=trainable)
        if collection:
            tf.add_to_collection(collection, ker)

        # Apply the conv layer
        conv = tf.nn.conv2d(inputs, ker, strides=[1, 1, 1, 1], padding=padding)

        if batch_norm:
            pre_act = batch_norm_layer(conv, in_type='conv', collection=collection, training=trainable)
        else:
            b = tf.Variable(tf.constant(0.01, shape=[filters]), name="Bias", trainable=trainable)
            if collection:
                tf.add_to_collection(collection, b)
            pre_act = tf.nn.bias_add(conv, b)

        act = activation(pre_act) if activation is not None else pre_act

        return act


def fc_layer(inputs, units, activation=tf.nn.relu,
             wd=None, trainable=True, use_bias=True,
             batch_norm=False, collection=None, name="fc"):
    """Wraps a fully connected layer with bias, regularization, normalization"""

    with tf.variable_scope(name):
        # flatten the input to [batch, dim]
        in_shape = inputs.get_shape().as_list()
        if len(in_shape) > 2:
            inputs = tf.reshape(inputs, [-1, np.prod(in_shape[1:])])
            in_shape = inputs.get_shape().as_list()

        w = _variable_with_weight_decay(name="Weights",
                                        shape=[in_shape[1], units],
                                        initializer=weight_var_init(in_shape[1]),
                                        weight_decay=wd,
                                        trainable=trainable)

        if collection:
            tf.add_to_collection(collection, w)

        # Apply the fc layer
        pre_act = tf.matmul(inputs, w)

        if batch_norm:
            pre_act = batch_norm_layer(pre_act, in_type='fc', collection=collection, training=trainable)
        elif use_bias:
            b = tf.Variable(tf.constant(0.01, shape=[units]), name="Bias", trainable=trainable)
            if collection:
                tf.add_to_collection(collection, b)
            pre_act = tf.nn.bias_add(pre_act, b)

        act = activation(pre_act) if activation is not None else pre_act

        return act


def batch_norm_layer(inputs, in_type='conv', collection=None, training=True):
    """
    Batch normalization layer that maintains moving averages of
    mean and var.
    in_type: one of 'conv' or 'fc'
    """

    # Configuration parameters not currently exposed
    name = "batch_norm"
    avg_decay = 0.99
    scale = True and training
    offset = True and training

    # Select the dimensions to normalize based on input type
    if in_type.lower() == 'conv':
        dims = [0, 1, 2]
    elif in_type.lower() == 'fc':
        dims = [0]
    else:
        raise ValueError

    with tf.variable_scope(name):
        channels = inputs.get_shape().as_list()[-1]
        # initializers
        gamma = tf.Variable(tf.constant(1.0, shape=[channels]), name="Gamma", trainable=scale)
        beta = tf.Variable(tf.constant(0.0, shape=[channels]), name="Beta", trainable=offset)
        mean_avg = tf.Variable(tf.constant(0.0, shape=[channels]), name="Mean_avg", trainable=False)
        var_avg = tf.Variable(tf.constant(1.0, shape=[channels]), name="Var_avg", trainable=False)
        eps = 0.001

        if collection:
            tf.add_to_collection(collection, gamma)
            tf.add_to_collection(collection, beta)
            tf.add_to_collection(collection, mean_avg)
            tf.add_to_collection(collection, var_avg)

        if training:
            mean, var = tf.nn.moments(inputs, axes=dims, keep_dims=False)

            # update averages
            mean_avg = mean_avg.assign_sub((1 - avg_decay) * tf.subtract(mean_avg, mean))
            var_avg = var_avg.assign_sub((1 - avg_decay) * tf.subtract(var_avg, var))
        else:
            mean, var = mean_avg, var_avg

        avg_op = tf.group(mean_avg, var_avg) if training else tf.no_op()

        with tf.control_dependencies([avg_op]):
            normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, eps)

        # do summary
        tf.summary.histogram("activations", normed)
        if training:
            tf.summary.histogram("Means", mean)
            tf.summary.histogram("Vars", var)
            tf.summary.histogram("Mean_avg", mean_avg)
            tf.summary.histogram("Var_avg", var_avg)

        return normed
