
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from typing import Any, Mapping

from ncinet.ncinet_input import training_inputs
from .model import NciKeys
from .config_meta import SessionConfig, EvalConfig


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
            model_ops = tf.get_collection(NciKeys.AE_ENCODER_VARIABLES) \
                        + tf.get_collection(NciKeys.AE_DECODER_VARIABLES)
        else:
            model_ops = tf.get_collection(NciKeys.AE_ENCODER_VARIABLES) \
                        + tf.get_collection(NciKeys.INF_VARIABLES)

        saver = tf.train.Saver(model_ops)

        scaffold = dict(init_fn=lambda scaffold, sess: load_trained(scaffold['saver'], sess),
                        ready_op=tf.report_uninitialized_variables(),
                        summary_writer=summary_writer,
                        summary_op=summary_op,
                        saver=saver)

        return scaffold


def eval_once(scaffold, eval_op, sess_config):
    # type: (Mapping[str, Any], tf.Tensor, SessionConfig) -> Mapping[str, float]
    """Run an operation once on the eval data.
    Args:
        scaffold: dict which approximates a tf.train.Scaffold
        eval_op: op to run on eval data.
        sess_config: initialized SessionConfig
    """
    config = sess_config.eval_config
    label_type = sess_config.model_config.label_type

    with tf.Session() as sess:
        # initialize the session.
        global_step = scaffold['init_fn'](scaffold, sess)

        # check if session is ready.
        not_init = sess.run(scaffold['ready_op'])
        if len(not_init) != 0:
            print(not_init)
            raise RuntimeError

        # runtime parameters
        batch_gen = training_inputs(eval_data=config.use_eval_data, batch_size=config.batch_size,
                                    request=sess_config.request, ingest_config=sess_config.ingest_config,
                                    data_types=('names', 'fingerprints', label_type), repeat=False)

        total_sample_count = len(batch_gen)
        num_iter = int(math.ceil(total_sample_count / config.batch_size))
        step = 0

        # accumulator for eval op.
        eval_acc = 0

        # Initialize data writer
        data_writer = config.data_writer
        data_writer.setup(sess)

        while step < num_iter:
            name_batch, print_batch, label_batch = next(batch_gen)
            print_batch = list(print_batch)

            eval_val, *data_writer.data_ops = sess.run([eval_op] + data_writer.data_ops,
                                                       feed_dict={'prints:0': print_batch,
                                                                  'labels:0': label_batch})

            # Collect batch data
            data_writer.collect_batch((eval_val, name_batch, print_batch, label_batch))

            eval_acc += np.sum(eval_val)
            step += 1

        # summary steps
        summary = tf.Summary()

        results = {}
        if sess_config.model_config.is_autoencoder:
            avg_error = eval_acc / total_sample_count
            print("{}; step {}: average per-pixel error {:.3f}".format(datetime.now(), global_step, avg_error))
            results['error'] = avg_error
        else:
            # Compute precision @ 1.
            precision = eval_acc / total_sample_count
            print('{}; {}: precision @ 1 = {:.3f}'.format(datetime.now(), global_step, precision))
            results['precision'] = precision

        # add results to the summary
        for tag, value in results.items():
            summary.value.add(tag=tag, simple_value=value)

        scaffold['summary_writer'].add_summary(summary, global_step)

        # save the recorded data
        data_writer.save()

        return results


def evaluate(config):
    # type: (SessionConfig) -> Mapping[str, float]
    """Eval model for a number of steps."""
    with tf.Graph().as_default() as g:
        # Construct computation graph
        logits = config.logits_network_gen(g, config.model_config, eval_net=True)
        labels = config.labels_network_gen(g, eval_net=True)
 
        # Build the eval operations.
        eval_op = config.eval_metric(logits, labels)

        # Construct helpers to run model.
        scaffold = _make_scaffold(g, config.eval_config, config.model_config.is_autoencoder)

        while True:
            eval_result = eval_once(scaffold, eval_op, config)
            if config.eval_config.run_once:
                return eval_result
            time.sleep(config.eval_config.eval_interval)


def main(config):
    if tf.gfile.Exists(config.eval_config.eval_dir):
        tf.gfile.DeleteRecursively(config.eval_config.eval_dir)
        tf.gfile.MakeDirs(config.eval_config.eval_dir)

    return evaluate(config)
