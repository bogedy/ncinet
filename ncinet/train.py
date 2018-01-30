
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import ncinet.ncinet_input
import ncinet.model_train

from ncinet.model import NciKeys
from . import WORK_DIR
from . import training_config
from .config_meta import ModelConfig, SessionConfig


class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""
    def __init__(self, loss_op):
        self._loss_op = loss_op
        self._step = -1
        self._start_time = time.time()

    def begin(self):
        self._step = -1
        self._start_time = time.time()

    def before_run(self, run_context):
        self._step += 1
        if self._step == 0:
            run_args = None
        else:
            run_args = tf.train.SessionRunArgs(self._loss_op)  # Asks for loss value.
        return run_args

    def after_run(self, run_context, run_values):
        if self._step > 0:
            if self._step % training_config.log_frequency == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results
                examples_per_sec = training_config.log_frequency * training_config.batch_size / duration
                sec_per_batch = float(duration / training_config.log_frequency)

                format_str = ('{date}: step {step}, loss = {loss:.2} ({eps:.1} '
                              'examples/sec; {spb:.3} sec/batch)')
                print(format_str.format(date=datetime.now(), step=self._step, loss=loss_value,
                                        eps=examples_per_sec, spb=sec_per_batch))


def _make_scaffold(graph):
    # type: (tf.Graph) -> tf.train.Scaffold
    """Constructs a scaffold for training"""
    with graph.as_default():
        summary = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        ready_op = tf.report_uninitialized_variables()

        if TRAIN_AUTOENCODER:
            saver_auto = tf.train.Saver(tf.get_collection(NciKeys.AE_ENCODER_VARIABLES)
                                        + tf.get_collection(NciKeys.AE_DECODER_VARIABLES))
            scaffold = tf.train.Scaffold(init_op=init_op, ready_op=ready_op,
                                         summary_op=summary,
                                         saver=saver_auto)
        else:
            saver_auto = tf.train.Saver(tf.get_collection(NciKeys.AE_ENCODER_VARIABLES))
            saver_inf = tf.train.Saver(tf.get_collection(NciKeys.AE_ENCODER_VARIABLES)
                                       + tf.get_collection(NciKeys.INF_VARIABLES))

            def load_trained(_, sess):
                """Load weights from a trained model"""
                # restore vars
                ckpt = tf.train.get_checkpoint_state(AE_TRAIN_DIR)
                if ckpt and ckpt.model_checkpoint_path:
                    # Restores from checkpoint
                    with tf.variable_scope(tf.get_variable_scope()):
                        saver_auto.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print('No checkpoint file found')
                    raise RuntimeError

            scaffold = tf.train.Scaffold(init_op=init_op,
                                         init_fn=load_trained,
                                         ready_op=ready_op,
                                         summary_op=summary, saver=saver_inf)

        return scaffold


def train(config):
    # type: (SessionConfig) -> None
    with tf.Graph().as_default() as g:
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Make graph for main network
        logits, labels = config.logits_network_gen(g, config.model_config)

        # Calculate loss
        loss = ncinet.model_train.loss(logits, labels, xent_type=config.xent_type)

        # build training operation
        train_op = ncinet.model_train.train(loss, global_step)

        # Generate batches of inputs and labels
        batch_gen = config.batch_gen()

        check = tf.add_check_numerics_ops()
        scaffold = _make_scaffold(g)

        # Run the training on the graph
        with tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            checkpoint_dir=training_config.train_dir,
            save_summaries_steps=training_config.summary_steps,
            save_checkpoint_secs=training_config.checkpoint_secs,
            hooks=[tf.train.StopAtStepHook(num_steps=training_config.max_steps),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook(loss)]) as mon_sess:

            while not mon_sess.should_stop():
                print_batch, label_batch = next(batch_gen)

                # Run the graph once
                print_batch = list(print_batch)
                mon_sess.run([train_op, check],
                             feed_dict={'prints:0': print_batch,
                                        'labels:0': label_batch})


# TODO: remove these globals
def main(options):
    global TRAIN_AUTOENCODER
    global INF_TYPE
    global AE_TRAIN_DIR

    TRAIN_AUTOENCODER = (options.model == 'AE')
    INF_TYPE = options.model

    AE_TRAIN_DIR = os.path.join(WORK_DIR, "train_ae")
    inf_train_dir = os.path.join(WORK_DIR, "train_inf_" + INF_TYPE)

    training_config.train_dir = AE_TRAIN_DIR if TRAIN_AUTOENCODER else inf_train_dir

    # Delete any existing training files
    if tf.gfile.Exists(training_config.train_dir):
        tf.gfile.DeleteRecursively(training_config.train_dir)
        tf.gfile.MakeDirs(training_config.train_dir)

    from .config_init import TopoSessionConfig, EncoderSessionConfig, SignSessionConfig

    if options.model == 'AE':
        config = EncoderSessionConfig()
    elif options.model == 'topo':
        config = TopoSessionConfig()
    elif options.model == 'sign':
        config = SignSessionConfig()
    else:
        raise ValueError

    train(config)
