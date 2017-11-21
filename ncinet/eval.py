
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import model
import ncinet_input


WORK_DIR = model.WORK_DIR

BATCH_SIZE = 100
NUM_EVAL = 600
USE_EVAL_DATA = True

EVAL_AUTOENCODER = True
EVAL_DIR = os.path.join(WORK_DIR, "eval_ae" if EVAL_AUTOENCODER else "eval_inf")
AUTO_DIR = os.path.join(WORK_DIR, "train_ae")
INF_DIR = os.path.join(WORK_DIR, "train_inf")
TRAIN_DIR = AUTO_DIR if EVAL_AUTOENCODER else INF_DIR

RUN_ONCE = True
EVAL_INTERVAL = 120


def _make_scaffold(graph, autoencoder=True):
    """Construct a 'scaffold' for the given model"""
    with graph.as_default():
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(EVAL_DIR, graph)

        def load_trained(saver, sess):
            # restore vars
            ckpt = tf.train.get_checkpoint_state(TRAIN_DIR)
            if ckpt and ckpt.model_checkpoint_path:
                # extract global_step from checkpoint filename
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                # Restores from checkpoint
                with tf.variable_scope(tf.get_variable_scope()):
                    saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('No checkpoint file found')
                raise RuntimeError

            return global_step

        if autoencoder:
            saver = tf.train.Saver(tf.get_collection(model.NciKeys.AE_ENCODER_VARIABLES)
                                   + tf.get_collection(model.NciKeys.AE_DECODER_VARIABLES))
        else:
            saver = tf.train.Saver(tf.get_collection(model.NciKeys.AE_ENCODER_VARIABLES)
                                   + tf.get_collection(model.NciKeys.INF_VARIABLES))

        scaffold = dict(init_fn=lambda scaffold, sess: load_trained(scaffold['saver'], sess),
                        ready_op=tf.report_uninitialized_variables(),
                        summary_writer=summary_writer,
                        summary_op=summary_op,
                        saver=saver)

        return scaffold


def eval_once(scaffold, eval_op):
    """Run an operation once on the eval data.
    Args:
        scaffold: dict which approximates a tf.train.Scaffold
        eval_op: op to run on eval data.
    """
    with tf.Session() as sess:

        # initialize the session.
        global_step = scaffold['init_fn'](scaffold, sess)

        # check if session is ready.
        not_init = sess.run(scaffold['ready_op'])
        if len(not_init) != 0:
            print(not_init)
            raise RuntimeError

        # runtime parameters
        num_iter = int(math.ceil(NUM_EVAL / BATCH_SIZE))
        total_sample_count = num_iter * BATCH_SIZE
        step = 0

        # accumulator for eval op.
        eval_acc = 0

        batch_gen = ncinet_input.inputs(USE_EVAL_DATA, BATCH_SIZE, data_types=['fingerprints', 'topologies'])

        while step < num_iter:
            print_batch, topo_batch = next(batch_gen)
            print_batch = list(print_batch)

            if EVAL_AUTOENCODER:
                label_batch = print_batch
            else:
                label_batch = topo_batch

            eval_val = sess.run([eval_op],
                                feed_dict={'prints:0': print_batch,
                                           'labels:0': label_batch})

            eval_acc += eval_val[0]
            step += 1

        # summary steps
        summary = tf.Summary()
        summary.ParseFromString(sess.run(scaffold['summary_op'],
                                         feed_dict={'prints:0': print_batch}))

        if EVAL_AUTOENCODER:
            avg_error = eval_acc / total_sample_count
            print("{}: average per-pixel error {:.3f}".format(datetime.now(), avg_error))
            summary.value.add(tag='error', simple_value=avg_error)
        else:
            # Compute precision @ 1.
            precision = eval_acc / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            summary.value.add(tag='precision', simple_value=precision)

        scaffold['summary_writer'].add_summary(summary, global_step)

        #trained_vars = {}
        #for var in tf.trainable_variables()+tf.model_variables():
        #    trained_vars[var.op.name] = var.eval(sess)

        #with open(os.path.join(WORK_DIR, "vars.npz"), 'w') as vars_file:
        #    np.savez(vars_file, **trained_vars)


def evaluate():
    """Eval model for a number of steps."""
    with tf.Graph().as_default() as g:
        # Placeholders for nci prints and scores
        prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")
        if EVAL_AUTOENCODER:
            labels = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="labels")
        else:
            labels = tf.placeholder(tf.int32, shape=[None], name="labels")

        # Build a Graph to run the model
        if EVAL_AUTOENCODER:
            logits = model.autoencoder(prints, training=False)
        else:
            logits = model.inference(prints, training=False)

        # Build the eval operations.
        if EVAL_AUTOENCODER:
            # Calculate norm of difference.
            norms = tf.norm(tf.subtract(labels, logits), ord="fro", axis=[1, 2])
            eval_op = tf.divide(tf.reduce_sum(norms), 100*100)
        else:
            # Calculate precision @1.
            top_k = tf.nn.in_top_k(logits, labels, 1)
            eval_op = tf.count_nonzero(top_k)

        # Construct helpers to run model.
        scaffold = _make_scaffold(g, EVAL_AUTOENCODER)

        while True:
            eval_once(scaffold, eval_op)
            if RUN_ONCE:
                break
            time.sleep(EVAL_INTERVAL)


def main(argv=None):
    if tf.gfile.Exists(EVAL_DIR):
        tf.gfile.DeleteRecursively(EVAL_DIR)
        tf.gfile.MakeDirs(EVAL_DIR)
    evaluate()


if __name__ == '__main__':
    main()
