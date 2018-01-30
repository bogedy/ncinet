"""
Collection of functions which set up graphs
"""

import tensorflow as tf

from .config_meta import EncoderConfig, InfConfig, SessionConfig
from .ncinet_input import inputs

from typing import Tuple

batch_size = 32


class EncoderSessionConfig(SessionConfig):
    batch_gen_args = {'eval_data': False, 'batch_size': batch_size, 'data_types': ['fingerprints']}
    xent_type = 'sigmoid'
    model_config = EncoderConfig()

    def logits_network_gen(self, graph, config):
        # type: (tf.Graph, EncoderConfig) -> Tuple[tf.Tensor, tf.Tensor]
        with graph.as_default():
            # Fingerprint placeholders
            prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")
            labels = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="labels")

            # Calculate logits and loss
            from .model import autoencoder
            logits = autoencoder(prints, config)

            return logits, labels

    def batch_gen(self):
        def add_noise(x, factor):
            import numpy as np
            noise = np.random.randn(*x.shape)
            x = x + factor * noise
            return np.clip(x, 0., 1.)

        batch_gen = inputs(**EncoderSessionConfig.batch_gen_args)

        def wrapped_gen():
            while True:
                prints = next(batch_gen)[0]
                labels = prints
                prints = add_noise(prints, 0.1)
                yield prints, labels

        return wrapped_gen()


class InfSessionConfig(SessionConfig):
    xent_type = 'softmax'
    inf_type = ''

    def logits_network_gen(self, graph, config):
        # type: (tf.Graph, InfConfig) -> Tuple[tf.Tensor, tf.Tensor]
        with graph.as_default():
            # Fingerprint placeholders
            prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")
            labels_input = tf.placeholder(tf.int32, shape=[None], name="labels")
            if self.inf_type == "topo":
                labels = tf.one_hot(labels_input, 4, dtype=tf.float32)
            elif self.inf_type == "sign":
                labels_index = tf.floordiv(tf.add(tf.cast(tf.sign(labels_input), tf.int32), 1), 2)
                labels = tf.one_hot(labels_index, 2, dtype=tf.float32)
            else:
                raise ValueError

            # Calculate logits and loss
            from .model import inf_classify
            logits = inf_classify(prints, config)

            return logits, labels

    def batch_gen(self):
        return inputs(**self.batch_gen_args)


class TopoSessionConfig(InfSessionConfig):
    inf_type = 'topo'
    batch_gen_args = {'eval_data': False, 'batch_size': batch_size, 'data_types': ['fingerprints', 'topologies']}
    model_config = InfConfig()


class SignSessionConfig(InfSessionConfig):
    inf_type = 'sign'
    batch_gen_args = {'eval_data': False, 'batch_size': batch_size, 'data_types': ['fingerprints', 'scores']}
    model_config = InfConfig(n_logits=2)
