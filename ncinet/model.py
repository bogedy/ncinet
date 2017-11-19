
# code adapted from tutorials:
# https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial/blob/master/mnist.py
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from layers import conv_layer, fc_layer, NciKeys

# TODO: consider binding some actions to CPU

# Directory where summaries and checkpoints are written.
#WORK_DIR = '/work/05187/ams13/maverick/Working/TensorFlow/'
WORK_DIR = 'C:/Users/schan/Documents/TF_run'

# TODO: these numbers are not correct. Should be set in ncinet_input
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 12000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 600

# TODO: learning rate decay
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.05  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.005       # Initial learning rate.
BATCH_SIZE = 32


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
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')

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


# main nn model
# TODO: update num logits (and output in general)
# TODO: generalize to allow hyperparameter optimization
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

