
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from typing import Any, Mapping

import ncinet.model
import ncinet.ncinet_input

from .model import NciKeys
from . import WORK_DIR
from .config_meta import SessionConfig, EvalConfig, EvalWriterBase
from .config_init import EncoderSessionConfig, TopoSessionConfig, SignSessionConfig, EvalWriter


def _make_scaffold(graph, config, autoencoder=True):
    # type: (tf.Graph, EvalConfig, bool) -> Mapping[str, Any]
    """Construct a 'scaffold' for the given model"""
    with graph.as_default():
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.eval_dir, graph)

        def load_trained(training_saver, sess):
            """Restores variables from training"""
            ckpt = tf.train.get_checkpoint_state(config.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # extract global_step from checkpoint filename
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                # Restores from checkpoint
                with tf.variable_scope(tf.get_variable_scope()):
                    training_saver.restore(sess, ckpt.model_checkpoint_path)
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


def eval_once(scaffold, eval_op, config):
    # type: (Mapping[str, Any], tf.Tensor, EvalConfig) -> None
    """Run an operation once on the eval data.
    Args:
        scaffold: dict which approximates a tf.train.Scaffold
        eval_op: op to run on eval data.
        config: initialized EvalConfig
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
        batch_gen = ncinet.ncinet_input.inputs(config.use_eval_data, config.batch_size,
                                               data_types=['names', 'fingerprints', 'topologies'],
                                               repeat=False)

        total_sample_count = len(batch_gen)
        num_iter = int(math.ceil(total_sample_count / config.batch_size))
        step = 0

        # accumulator for eval op.
        eval_acc = 0

        # Initialize data writer
        data_writer = config.data_writer
        data_writer.setup(sess)

        while step < num_iter:
            name_batch, print_batch, topo_batch = next(batch_gen)
            print_batch = list(print_batch)

            if EVAL_AUTOENCODER:
                label_batch = print_batch
            else:
                label_batch = topo_batch

            eval_val, *data_writer.data_ops = sess.run([eval_op] + data_writer.data_ops,
                                                       feed_dict={'prints:0': print_batch,
                                                                  'labels:0': label_batch})

            # Collect batch data
            data_writer.collect_batch((eval_val, name_batch, print_batch, topo_batch))

            eval_acc += np.sum(eval_val)
            step += 1

        # summary steps
        summary = tf.Summary()

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

        # save the recorded data
        data_writer.save()


def evaluate(config):
    # type: (SessionConfig) -> None
    """Eval model for a number of steps."""
    with tf.Graph().as_default() as g:
        # Construct computation graph
        logits, labels = config.logits_network_gen(g, config.model_config, eval_net=True)
 
        # Build the eval operations.
        eval_op = config.eval_metric(logits, labels)

        # Construct helpers to run model.
        scaffold = _make_scaffold(g, config.eval_config, EVAL_AUTOENCODER)

        while True:
            eval_once(scaffold, eval_op, config.eval_config)
            if config.eval_config.run_once:
                break
            time.sleep(config.eval_config.eval_interval)


# TODO: remove globals
def main(options):
    global EVAL_AUTOENCODER
    EVAL_AUTOENCODER = (options.model == 'AE')

    inf_type = options.model

    eval_dir = os.path.join(WORK_DIR, "eval_ae" if EVAL_AUTOENCODER else "eval_inf_" + inf_type)
    auto_dir = os.path.join(WORK_DIR, "train_ae")
    inf_dir = os.path.join(WORK_DIR, "train_inf_" + inf_type)
    train_dir = auto_dir if EVAL_AUTOENCODER else inf_dir

    run_once = True

    if not run_once:
        data_writer = EvalWriterBase()
    else:
        op_name = ("max_pooling2d_3/MaxPool:0",)
        file_name = "{}_results_{}".format('eval', 'ae' if EVAL_AUTOENCODER else 'inf')
        data_writer = EvalWriter(archive_name=file_name,
                                 archive_dir=WORK_DIR,
                                 saved_vars=op_name)

    eval_config = EvalConfig(batch_size=100,
                             eval_dir=eval_dir,
                             train_dir=train_dir,
                             run_once=run_once,
                             data_writer=data_writer)

    if options.model == 'AE':
        config = EncoderSessionConfig(eval_config=eval_config)
    elif options.model == 'topo':
        config = TopoSessionConfig(eval_config=eval_config)
    elif options.model == 'sign':
        config = SignSessionConfig(eval_config=eval_config)
    else:
        raise ValueError

    if tf.gfile.Exists(eval_dir):
        tf.gfile.DeleteRecursively(eval_dir)
        tf.gfile.MakeDirs(eval_dir)

    evaluate(config)
