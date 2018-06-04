"""
Constructs parts of the graph only used in training. This currently includes
the loss function and the training operation.
"""

import tensorflow as tf

from .config_meta import TrainingConfig

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.


def loss(logits, labels, xent_type="softmax"):
    """Sums L2Loss for trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits: Tensor of unscaled logits.
        labels: Tensor of labels. Shape should match logits.
        xent_type: type of loss to perform ("softmax" or "sigmoid")
    Returns:
        Loss tensor of type float.
    """
    # type: (tf.Tensor, tf.Tensor, str) -> tf.Tensor
    if xent_type != "softmax" and xent_type != "sigmoid":
        raise ValueError

    # record labels
    tf.summary.histogram("labels", labels)
    tf.summary.histogram("logits", tf.nn.sigmoid(logits))

    # batch entropy
    ent_f = tf.nn.softmax_cross_entropy_with_logits if xent_type == 'softmax' \
        else tf.nn.sigmoid_cross_entropy_with_logits
    x_ent = ent_f(logits=logits, labels=labels, name="entropy_per_ex")
    x_ent_mean = tf.reduce_mean(x_ent, name="cross_entropy")
    tf.add_to_collection('losses', x_ent_mean)

    # The total loss is the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss: Total loss from loss().
    Returns:
        loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='loss_avg')
    losses = tf.get_collection('losses')

    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Write summary for total loss
    tf.summary.scalar("Total loss", total_loss)
    tf.summary.scalar("Total loss (avg)", loss_averages.average(total_loss))

    # Attach a scalar summary to all individual losses and averages.
    for l in losses:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name + ' (avg)', loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step, config):
    # type: (tf.Tensor, tf.Tensor, TrainingConfig) -> tf.Tensor
    """Build an op to run training.

    Calculate the learning rate, create an optimizer and apply to all
    trainable variables.
    Args:
        total_loss: Total loss from loss().
        global_step: Integer Variable counting the number of training steps
          processed.
        config: Config object containing training hyperparameters
    Returns:
        train_op: op for training.
    """

    # Variables that affect learning rate.
    num_batches_per_epoch = config.num_examples_per_epoch_train / config.batch_size
    decay_steps = int(num_batches_per_epoch * config.num_epochs_per_decay)

    if config.use_learning_rate_decay:
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(config.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        config.learning_rate_decay_factor)
    else:
        lr = tf.constant(config.initial_learning_rate, name="learning_rate")

    tf.summary.scalar('learning_rate', lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op
