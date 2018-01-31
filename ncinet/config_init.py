"""
Model specific constructor functions.
"""

import tensorflow as tf

from .config_meta import SessionConfig
from .config_hyper import EncoderConfig, InfConfig
from .ncinet_input import inputs

from typing import Tuple


class EncoderSessionConfig(SessionConfig):
    xent_type = 'sigmoid'
    model_config = EncoderConfig()

    @property
    def batch_gen_args(self):
        return {'eval_data': False, 'batch_size': self.train_config.batch_size, 'data_types': ['fingerprints']}

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

        batch_gen = inputs(**self.batch_gen_args)

        def wrapped_gen():
            while True:
                prints = next(batch_gen)[0]
                labels = prints
                prints = add_noise(prints, 0.1)
                yield prints, labels

        return wrapped_gen()


class InfSessionConfig(SessionConfig):
    xent_type = 'softmax'
    inf_type = None             # type: str

    def logits_network_gen(self, graph, config):
        # type: (tf.Graph, InfConfig) -> Tuple[tf.Tensor, tf.Tensor]
        with graph.as_default():
            # Fingerprint placeholders
            prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")

            # Placeholders and preprocessing for labels.
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
    model_config = InfConfig()

    @property
    def batch_gen_args(self):
        return {'eval_data': False, 'batch_size': self.train_config.batch_size,
                'data_types': ['fingerprints', 'topologies']}


class SignSessionConfig(InfSessionConfig):
    inf_type = 'sign'
    model_config = InfConfig(n_logits=2)

    @property
    def batch_gen_args(self):
        return {'eval_data': False, 'batch_size': self.train_config.batch_size,
                'data_types': ['fingerprints', 'scores']}
