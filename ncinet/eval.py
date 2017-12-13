
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import ncinet.model
import ncinet.ncinet_input

from .model import NciKeys, WORK_DIR


BATCH_SIZE = 100
USE_EVAL_DATA = True

RUN_ONCE = False
WRITE_DATA = False
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
            saver = tf.train.Saver(tf.get_collection(NciKeys.AE_ENCODER_VARIABLES)
                                   + tf.get_collection(NciKeys.AE_DECODER_VARIABLES))
        else:
            saver = tf.train.Saver(tf.get_collection(NciKeys.AE_ENCODER_VARIABLES)
                                   + tf.get_collection(NciKeys.INF_VARIABLES))

        scaffold = dict(init_fn=lambda scaffold, sess: load_trained(scaffold['saver'], sess),
                        ready_op=tf.report_uninitialized_variables(),
                        summary_writer=summary_writer,
                        summary_op=summary_op,
                        saver=saver)

        return scaffold


def eval_once(scaffold, eval_op, topo=None):
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
        batch_gen = ncinet.ncinet_input.inputs(USE_EVAL_DATA, BATCH_SIZE,
                                               data_types=['names', 'fingerprints', 'topologies'],
                                               repeat=False,
                                               topo=topo)

        total_sample_count = len(batch_gen)
        num_iter = int(math.ceil(total_sample_count / BATCH_SIZE))
        step = 0

        # accumulator for eval op.
        eval_acc = 0

        # TODO: provide switches for these ops
        data_ops = []
        if WRITE_DATA:
            input_acc = []
            data_acc = []
            op_name = "fc1/Relu:0" if not EVAL_AUTOENCODER else "encoded/MaxPool:0"
            activations_op = sess.graph.get_tensor_by_name(op_name)
            data_ops.append(activations_op)

        while step < num_iter:
            name_batch, print_batch, topo_batch = next(batch_gen)
            print_batch = list(print_batch)

            if EVAL_AUTOENCODER:
                label_batch = print_batch
            else:
                label_batch = topo_batch

            eval_val, *data_val = sess.run([eval_op, *data_ops],
                                  feed_dict={'prints:0': print_batch,
                                             'labels:0': label_batch})

            # Collect other data
            if WRITE_DATA:
                input_acc.append((name_batch, print_batch, topo_batch))
                data_acc.append(data_val)

            eval_acc += eval_val
            step += 1

        # summary steps
        summary = tf.Summary()
        #all_eval_data = list(next(ncinet_input.inputs(USE_EVAL_DATA, total_sample_count, ['fingerprints']))[0])
        #summary.ParseFromString(sess.run(scaffold['summary_op'],
        #                                 feed_dict={'prints:0': all_eval_data}))

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

        # consolidate and save the recorded data
        if WRITE_DATA:
            if USE_EVAL_DATA:
                file_name = "eval_results_ae.npz" if EVAL_AUTOENCODER else "eval_results_inf.npz"
            else:
                file_name = "train_results_ae.npz" if EVAL_AUTOENCODER else "train_results_inf.npz"
            inputs = map(np.concatenate, zip(*input_acc))
            data = map(np.concatenate, zip(*data_acc))
            titles = ['names', 'fingerprints', 'topologies', 'activations']
            results = dict(zip(titles, list(inputs) + list(data)))

            with open(os.path.join(WORK_DIR, file_name), 'wb') as result_file:
                np.savez(result_file, **results)

        #trained_vars = {}
        #for var in tf.trainable_variables()+tf.model_variables():
        #    trained_vars[var.op.name] = var.eval(sess)

        #with open(os.path.join(WORK_DIR, "vars.npz"), 'w') as vars_file:
        #    np.savez(vars_file, **trained_vars)


def evaluate(topo=None):
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
            logits = ncinet.model.autoencoder(prints, training=False)
        else:
            if INF_TYPE == "topo":
                logits = ncinet.model.topo_classify(prints, training=False)
            elif INF_TYPE == "sign":
                logits = ncinet.model.sign_classify(prints, training=False)
            else:
                raise ValueError
 
        # Build the eval operations.
        if EVAL_AUTOENCODER:
            # Calculate norm of difference.
            norms = tf.norm(tf.subtract(labels, logits), ord="fro", axis=[1, 2])
            eval_op = tf.divide(tf.reduce_sum(norms), 100*100)
        else:
            if INF_TYPE == "sign":
                labels = tf.floordiv(tf.add(tf.cast(tf.sign(labels), tf.int32), 1), 2)

            # Calculate precision @1.
            top_k = tf.nn.in_top_k(logits, labels, 1)
            eval_op = tf.count_nonzero(top_k)

        # Construct helpers to run model.
        scaffold = _make_scaffold(g, EVAL_AUTOENCODER)

        while True:
            eval_once(scaffold, eval_op, topo=topo)
            if RUN_ONCE:
                break
            time.sleep(EVAL_INTERVAL)


# TODO: remove globals
def main(options):
    global EVAL_AUTOENCODER
    global INF_TYPE

    EVAL_AUTOENCODER = (options.model == 'AE')
    INF_TYPE = options.model

    global EVAL_DIR
    global TRAIN_DIR

    EVAL_DIR = os.path.join(WORK_DIR, "eval_ae" if EVAL_AUTOENCODER else "eval_inf_" + INF_TYPE)
    AUTO_DIR = os.path.join(WORK_DIR, "train_ae")
    INF_DIR = os.path.join(WORK_DIR, "train_inf_" + INF_TYPE)
    TRAIN_DIR = AUTO_DIR if EVAL_AUTOENCODER else INF_DIR

    if tf.gfile.Exists(EVAL_DIR):
        tf.gfile.DeleteRecursively(EVAL_DIR)
        tf.gfile.MakeDirs(EVAL_DIR)
    evaluate(topo=options.topo_restrict)


#if __name__ == '__main__':
#    main()
