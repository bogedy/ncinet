
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import ncinet.model
import ncinet.ncinet_input
import ncinet.model_train

from ncinet.model import NciKeys, WORK_DIR

# TODO: streamline the differences between training different models

LOG_FREQUENCY = 20
BATCH_SIZE = 32
TRAIN_DIR = ""
MAX_STEPS = 100000


class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime."""
    def __init__(self, loss_op):
        self._loss_op = loss_op

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
            if self._step % LOG_FREQUENCY == 0:
                current_time = time.time()
                duration = current_time - self._start_time
                self._start_time = current_time

                loss_value = run_values.results
                examples_per_sec = LOG_FREQUENCY * BATCH_SIZE / duration
                sec_per_batch = float(duration / LOG_FREQUENCY)

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), self._step, loss_value,
                      examples_per_sec, sec_per_batch))


def _make_scaffold(graph):
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

            def load_trained(scaffold, sess):
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


def train(topo=None):
    with tf.Graph().as_default() as g:
        global_step = tf.contrib.framework.get_or_create_global_step()

        #input placeholders
        prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")

        if TRAIN_AUTOENCODER:
            labels_input = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="labels")
            labels = labels_input
        else:
            labels_input = tf.placeholder(tf.int32, shape=[None], name="labels")
            if INF_TYPE == "topo":
                labels = tf.one_hot(labels_input, 4, dtype=tf.float32)
            elif INF_TYPE == "sign":
                labels_index = tf.floordiv(tf.add(tf.cast(tf.sign(labels_input), tf.int32), 1), 2)
                labels = tf.one_hot(labels_index, 2, dtype=tf.float32)
            else:
                raise ValueError

        # apply the nn
        if TRAIN_AUTOENCODER:
            logits = ncinet.model.autoencoder(prints)
        else:
            if INF_TYPE == "topo":
                logits = ncinet.model.inference(prints)
            elif INF_TYPE == "sign":
                logits = ncinet.model.sign_classify(prints)
            else:
                raise ValueError

        # calculate loss
        xent_type = 'sigmoid' if TRAIN_AUTOENCODER else 'softmax'
        loss = ncinet.model_train.loss(logits, labels, xent_type=xent_type)

        # build training operation
        train_op = ncinet.model_train.train(loss, global_step)

        # Set up framework to run model.
        if TRAIN_AUTOENCODER:
            in_args = {'eval_data': False, 'batch_size': BATCH_SIZE, 'data_types': ['fingerprints']}
        else:
            label_name = "topologies" if INF_TYPE == "topo" else "scores"
            in_args = {'eval_data': False, 'batch_size': BATCH_SIZE, 'data_types': ['fingerprints', label_name]}

        batch_gen = ncinet.ncinet_input.inputs(topo=topo, **in_args)

        check = tf.add_check_numerics_ops()
        scaffold = _make_scaffold(g)

        def add_noise(x, factor):
            noise = np.random.randn(*x.shape)
            x = x + factor*noise
            return np.clip(x, 0., 1.)

        with tf.train.MonitoredTrainingSession(
            scaffold=scaffold,
            checkpoint_dir=TRAIN_DIR,
            save_summaries_steps=50,
            save_checkpoint_secs=100,
            hooks=[tf.train.StopAtStepHook(num_steps=MAX_STEPS),
                   tf.train.NanTensorHook(loss),
                   _LoggerHook(loss)]) as mon_sess:

            while not mon_sess.should_stop():
                if TRAIN_AUTOENCODER:
                    print_batch = next(batch_gen)[0]
                    label_batch = print_batch
                    print_batch = add_noise(print_batch, 0.1)
                else:
                    print_batch, label_batch = next(batch_gen)

                print_batch = list(print_batch)

                mon_sess.run([train_op, check],
                             feed_dict={prints: print_batch,
                                        labels_input: label_batch})


# TODO: remove these globals
def main(options):
    global TRAIN_DIR
    global TRAIN_AUTOENCODER
    global INF_TYPE

    TRAIN_AUTOENCODER = (options.model == 'AE')
    INF_TYPE = options.model

    global AE_TRAIN_DIR
    AE_TRAIN_DIR = os.path.join(WORK_DIR, "train_ae")
    global INF_TRAIN_DIR
    INF_TRAIN_DIR = os.path.join(WORK_DIR, "train_inf_" + INF_TYPE)

    TRAIN_DIR = AE_TRAIN_DIR if TRAIN_AUTOENCODER else INF_TRAIN_DIR
    if tf.gfile.Exists(TRAIN_DIR):
        tf.gfile.DeleteRecursively(TRAIN_DIR)
        tf.gfile.MakeDirs(TRAIN_DIR)
    train(topo=options.topo_restrict)

#if __name__ == "__main__":
#    main()
