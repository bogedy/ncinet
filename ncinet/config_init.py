"""
Collection of functions which set up graphs
"""

import tensorflow as tf

from .config_meta import EncoderConfig, InfConfig, SessionConfig

from typing import Tuple


def _make_ae_graph(cls, graph, config):
    # type: (tf.Graph, EncoderConfig) -> Tuple[tf.Tensor, tf.Tensor]
    with graph.as_default():
        # Fingerprint placeholders
        prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")
        labels = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="labels")

        # Calculate logits and loss
        from .model import autoencoder
        logits = autoencoder(prints, config)

        return logits, labels


def _make_inf_graph(inf_type, graph, config):
    # type: (str, tf.Graph, InfConfig) -> Tuple[tf.Tensor, tf.Tensor]
    with graph.as_default():
        # Fingerprint placeholders
        prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")
        labels_input = tf.placeholder(tf.int32, shape=[None], name="labels")
        if inf_type == "topo":
            labels = tf.one_hot(labels_input, 4, dtype=tf.float32)
        elif inf_type == "sign":
            labels_index = tf.floordiv(tf.add(tf.cast(tf.sign(labels_input), tf.int32), 1), 2)
            labels = tf.one_hot(labels_index, 2, dtype=tf.float32)
        else:
            raise ValueError

        # Calculate logits and loss
        from .model import inf_classify
        logits = inf_classify(prints, config)

        return logits, labels


from .ncinet_input import inputs


def _ae_batch(**batch_gen_args):
    def add_noise(x, factor):
        import numpy as np
        noise = np.random.randn(*x.shape)
        x = x + factor * noise
        return np.clip(x, 0., 1.)

    batch_gen = inputs(**batch_gen_args)

    def wrapped_gen():
        while True:
            prints = next(batch_gen)[0]
            labels = prints
            prints = add_noise(prints, 0.1)
            yield prints, labels

    return wrapped_gen()

batch_size = 32

class EncoderSessionConfig(SessionConfig):
    logits_network_gen = _make_ae_graph
    batch_gen_args = {'eval_data': False, 'batch_size': batch_size, 'data_types': ['fingerprints']}
    xent_type = 'sigmoid'
    batch_gen = _ae_batch(**batch_gen_args)
    model_config = EncoderConfig()


class TopoSessionConfig(SessionConfig):
    logits_network_gen = lambda cls, graph, config: _make_inf_graph('topo', graph, config)
    batch_gen_args = {'eval_data': False, 'batch_size': batch_size, 'data_types': ['fingerprints', 'topologies']}
    xent_type = 'softmax'
    batch_gen = inputs(**batch_gen_args)
    model_config = InfConfig()


class SignSessionConfig(SessionConfig):
    logits_network_gen = lambda cls, graph, config: _make_inf_graph('sign', graph, config)
    batch_gen_args = {'eval_data': False, 'batch_size': batch_size, 'data_types': ['fingerprints', 'scores']}
    xent_type = 'sigmoid'
    batch_gen = inputs(**batch_gen_args)
    model_config = InfConfig(n_logits=2)
