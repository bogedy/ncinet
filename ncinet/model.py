
# code adapted from tutorials:
# https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

from __future__ import division

import math
import numpy as np
import tensorflow as tf

# TODO: consider binding some actions to CPU
# TODO: weights with biases

# Global constants describing the CIFAR-10 data set.
#IMAGE_SIZE = cifar10_input.IMAGE_SIZE
#NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 12000#cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 600 #cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.05  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.005       # Initial learning rate.
BATCH_SIZE = 32

class NciKeys:
    AE_ENCODER_VARIABLES="_NciKeys_ae_encoder_var"
    AE_DECODER_VARIABLES="_NciKeys_ae_decoder_var"
    INF_VARIABLES="_NciKeys_inf_var"


# TODO: factor this into layer wrappers?
def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    #tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))

def weight_var_init(dim):
    dim = tf.cast(dim, tf.float32)
    std_dev = tf.sqrt(2.0 / dim)
    return tf.truncated_normal_initializer(mean=0.0,
                                           stddev=std_dev,
                                           dtype=tf.float32)


def bias_var_init(shape):
    std_dev = math.sqrt(3.0 / shape[0])
    return tf.truncated_normal_initializer(mean=0.0,
                                           stddev=std_dev,
                                           dtype=tf.float32)


def _variable_with_weight_decay(name, shape, initializer, weight_decay, trainable=True):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """
    var = tf.get_variable(name,
                          shape=shape,
                          initializer=initializer,
                          dtype=tf.float32,
                          trainable=trainable)

    if trainable and weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


# TODO: add weight decay terms to these wrappers
# wraps a conv layer with name information and write summary info
# TODO: make api consistent with tf.layer
def conv_layer(inputs, filters, kernel_size, name='conv',
               padding='SAME',
               activation=tf.nn.relu,
               wd=None,
               batch_norm=False,
               collection=None,
               training=True):

    with tf.variable_scope(name):
        channels = inputs.get_shape().as_list()[-1]
        shape = [kernel_size[0], kernel_size[1], channels, filters]
        ker = _variable_with_weight_decay(name="Kernel",
                                          shape=shape,
                                          initializer=weight_var_init(np.prod(shape[:-1])),
                                          weight_decay=wd,
                                          trainable=training)
        if collection:
            tf.add_to_collection(collection, ker)

        if not batch_norm:
            b = tf.Variable(tf.constant(0.01, shape=[filters]), name="Bias", trainable=training)
            if collection:
                tf.add_to_collection(collection, b)

        conv = tf.nn.conv2d(inputs, ker, strides=[1, 1, 1, 1], padding=padding)
        #act = tf.nn.relu(tf.nn.bias_add(conv, b))
        pre_act = tf.nn.bias_add(conv, b) if not batch_norm else batch_norm_layer(conv, collection=collection, training=training)
        act = activation(pre_act) if activation is not None else pre_act

        #tf.summary.histogram("weights", w)
        #tf.summary.histogram("biases", b)

        return act


# wrap fc layer
# TODO: batch norm
def fc_layer(inputs, units, activation=tf.nn.relu,
             wd=None, training=True, use_bias=True,
             batch_norm=False, collection=None, name="fc"):

    with tf.variable_scope(name):
        # flatten the input to [batch, dim]
        in_shape = inputs.get_shape().as_list()
        if len(in_shape) > 2:
            inputs = tf.reshape(inputs, [in_shape[1], -1])
            in_shape = inputs.get_shape().as_list()

        w = _variable_with_weight_decay(name="Weights",
                                        shape=[in_shape[1], units],
                                        initializer=weight_var_init(in_shape[1]),
                                        weight_decay=wd,
                                        trainable=training)

        if collection:
            tf.add_to_collection(collection, w)

        pre_act = tf.matmul(inputs, w)
        if use_bias and not batch_norm:
            b = tf.Variable(tf.constant(0.01, shape=[units]), name="Bias", trainable=training)
            if collection:
                tf.add_to_collection(collection, b)
            pre_act = tf.nn.bias_add(pre_act, b)

        act = activation(pre_act) if activation is not None else pre_act

        #tf.summary.histogram("weights", w)
        #tf.summary.histogram("biases", b)
        #tf.summary.histogram("activations", act)

        return act


def batch_norm_layer(inputs, collection=None, training=True):
    name = "batch_norm"
    avg_decay = 0.99
    dims = [0,1,2] # for conv
    #dims = [0] # for fc
    scale = True and training
    offset = True and training
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
            decay_rate = avg_decay
            mean_avg = mean_avg.assign_sub((1 - decay_rate) * (mean_avg - mean))
            var_avg = var_avg.assign_sub((1 - decay_rate) * (var_avg - var))
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


# base feature detecting layer. Used in both autoencoder and inference
def bottom_convnet(prints, training=True):
    ### Encoder
    conv1 = conv_layer(inputs=prints, filters=32, kernel_size=(5,5),
                       name="conv1", wd=0.001, batch_norm=True,
                       collection=NciKeys.AE_ENCODER_VARIABLES, training=training)
    _activation_summary(conv1)
    # Now 100x100x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 50x50x32

    conv2 = conv_layer(inputs=maxpool1, filters=32, kernel_size=(5,5),
                       name="conv2", wd=0.001, collection=NciKeys.AE_ENCODER_VARIABLES,
                       training=training)
    _activation_summary(conv2)
    # Now 50x50x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 25x25x32

    conv3 = conv_layer(inputs=maxpool2, filters=16, kernel_size=(5,5),
                       name="conv3", wd=0.001, collection=NciKeys.AE_ENCODER_VARIABLES,
                       training=training)
    _activation_summary(conv3)
    # Now 25x25x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2,2), strides=(2,2), padding='same')
    # Now 13x13x16

    return encoded


def top_convnet_autoenc(encoded):
    ### Decoder
    upsample1 = tf.image.resize_images(encoded, size=(25,25), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 25x25x16
    conv4 = conv_layer(inputs=upsample1,
                       filters=16,
                       kernel_size=(5,5),
                       name="convT1",
                       collection=NciKeys.AE_DECODER_VARIABLES)
    # Now 25x25x16
    upsample2 = tf.image.resize_images(conv4, size=(50,50), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 50x50x16
    conv5 = conv_layer(inputs=upsample2,
                       filters=32,
                       kernel_size=(5,5),
                       name="convT2",
                       collection=NciKeys.AE_DECODER_VARIABLES)
    # Now 50x50x32
    upsample3 = tf.image.resize_images(conv5, size=(100,100), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Now 100x100x32
    conv6 = conv_layer(inputs=upsample3,
                       filters=32,
                       kernel_size=(5,5),
                       name="ConvT3",
                       collection=NciKeys.AE_DECODER_VARIABLES)
    # Now 100x100x32

    logits = conv_layer(inputs=conv6,
                        filters=1,
                        kernel_size=(5,5),
                        activation=None,
                        name="ConvT4",
                        collection=NciKeys.AE_DECODER_VARIABLES)
    #Now 100x100x1

    return logits


def autoencoder(prints, training=True):
    tf.summary.image("fingerprints", prints)
    base = bottom_convnet(prints, training=training)
    predict = top_convnet_autoenc(base)
    tf.summary.image("predicts", tf.nn.sigmoid(predict))
    return predict


# main nn model
# TODO: update num logits (and output in general)
# TODO: generalize to allow hyperparameter optimization
def inference(prints, training=True):
    tf.summary.image("fingerprints", prints)
    base = bottom_convnet(prints, training=False)
    flat = tf.reshape(base, [-1, 13*13*16])
    _activation_summary(flat)
    fc1 = fc_layer(flat, 512, name="fc1", collection=NciKeys.INF_VARIABLES, training=training)
    _activation_summary(fc1)
    logits = fc_layer(fc1, 4, name="fc2", activation=None, collection=NciKeys.INF_VARIABLES, training=training)
    _activation_summary(logits)
    return logits


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Logits from inference().
        labels: Labels from distorted_inputs or inputs(). 1-D tensor
                of shape [batch_size]
    Returns:
        Loss tensor of type float.
    """

    # record labels
    tf.summary.histogram("labels", labels)
    tf.summary.histogram("logits", tf.nn.sigmoid(logits))

    # batch entropy
    x_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name="entropy_per_ex")
    x_ent_mean = tf.reduce_mean(x_ent, name="cross_entropy")
    tf.add_to_collection('losses', x_ent_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in CIFAR-10 model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')

    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name + ' (avg)', loss_averages.average(l))

    return loss_averages_op


# TODO: simplify this
def train(total_loss, global_step):
    """Train CIFAR-10 model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
    Returns:
        train_op: op for training.
    """

    # Variables that affect learning rate.
    #num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / BATCH_SIZE
    #decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    #lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
    #                                global_step,
    #                                decay_steps,
    #                                LEARNING_RATE_DECAY_FACTOR,
    #                                staircase=True)
    lr = tf.constant(INITIAL_LEARNING_RATE, name="learning_rate")
    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('loss', total_loss)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # clip gradients
    #grads_clip = [(tf.clip_by_value(g[0], -5., 5.), g[1]) for g in grads]

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    #variable_averages = tf.train.ExponentialMovingAverage(
    #    MOVING_AVERAGE_DECAY, global_step)
    #variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op]): #, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op

