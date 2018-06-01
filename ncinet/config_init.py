"""
Defines session configuration objects. These objects contain hyperparameter
and other data, and also implement functions that differ between models.
"""

import tensorflow as tf
import numpy as np

from .config_meta import SessionConfig, EvalWriterBase
from .config_hyper import EncoderConfig, InfConfig
from .ncinet_input import training_inputs

from typing import List, Tuple, Any


class EncoderSessionConfig(SessionConfig):
    """Session parameters for autoencoder network."""
    xent_type = 'sigmoid'
    model_config = EncoderConfig()

    def logits_network_gen(self, graph, config, eval_net=False):
        # type: (tf.Graph, EncoderConfig, bool) -> tf.Tensor
        """Generate autoencoder graph."""
        with graph.as_default():
            # Fingerprint placeholders
            prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")

            # Calculate logits and loss
            from .model import autoencoder
            logits = autoencoder(prints, config, training=(not eval_net))

            return logits

    def labels_network_gen(self, graph, eval_net=False):
        # type: (tf.Graph, bool) -> tf.Tensor
        """Construct placeholders for the fingerprints."""
        with graph.as_default():
            labels = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="labels")
            return labels

    def eval_metric(self, logits, labels):
        """Calculate norm of the difference of original and output."""
        logits = tf.nn.sigmoid(logits)
        norms = tf.norm(tf.subtract(labels, logits), ord="fro", axis=[1, 2])
        eval_op = tf.divide(norms, 100 * 100)
        # eval_op = tf.divide(tf.reduce_sum(norms), 100 * 100)
        return eval_op

    def batch_gen(self):
        """Generate batches of noised NCI plots for the autoencoder"""
        def add_noise(x, factor):
            """Add white noise to NCI input."""
            noise = np.random.randn(*x.shape)
            x = x + factor * noise
            return np.clip(x, 0., 1.)

        batch_gen = training_inputs(eval_data=False, batch_size=self.train_config.batch_size,
                                    request=self.request, ingest_config=self.ingest_config,
                                    data_types=('fingerprints',))

        def wrapped_gen():
            """Processes the output of the batch generator"""
            while True:
                prints = next(batch_gen)[0]
                labels = prints
                prints = add_noise(prints, self.train_config.input_noise)
                yield prints, labels

        return wrapped_gen()


class InfSessionConfig(SessionConfig):
    """Session parameters for inference network."""
    xent_type = 'softmax'
    model_config = None         # type: InfConfig

    def logits_network_gen(self, graph, config, eval_net=False):
        # type: (tf.Graph, InfConfig, bool) -> tf.Tensor
        """Constructs an encoder followed by an inference network."""
        with graph.as_default():
            # Fingerprint placeholders
            prints = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="prints")

            # Calculate logits
            from .model import inf_classify
            logits = inf_classify(prints, config, training=(not eval_net))

            return logits

    def labels_network_gen(self, graph, eval_net=False):
        # type: (tf.Graph, bool) -> tf.Tensor
        """Serve class labels as class indices or one-hot vectors as needed."""
        with graph.as_default():
            # Placeholders and preprocessing for labels.
            labels_input = tf.placeholder(tf.int32, shape=[None], name="labels")

            if not eval_net:
                labels = tf.one_hot(labels_input, self.model_config.n_logits, dtype=tf.float32)
            else:
                labels = labels_input

            return labels

    def eval_metric(self, logits, labels):
        """Calculate precision @1"""
        # Convert stability score to one_hot (x > 0)
        if self.inf_type == "sign":
            labels = tf.floordiv(tf.add(tf.sign(labels), 1), 2)

        labels = tf.cast(labels, tf.int32)
        top_k = tf.nn.in_top_k(logits, labels, 1)
        # eval_op = tf.count_nonzero(top_k)
        return top_k


class EvalWriter(EvalWriterBase):
    """Stores data from eval runs.

    Parameters
    ----------
    archive_name: str
        Name of the output archive.
    archive_dir: path
        Directory to write the output.
    saved_vars: Tuple[str, ...]
        List with the names of nodes whose activations will be recorded.
    """
    def __init__(self, archive_name, archive_dir, saved_vars):
        # type: (str, str, Tuple[str, ...]) -> None
        self.archive_name = archive_name
        self.archive_dir = archive_dir
        self.activation_names = saved_vars
        self.activation_ops = None      # type: List[tf.Tensor, ...]

        # Accumulators for captured data
        self.activation_acc = []
        self.inputs_acc = []

        # freeze the class
        EvalWriterBase.__init__(self)

    def setup(self, sess):
        """Setup writer using the the current session"""
        # type: tf.Session -> None
        self.activation_ops = [sess.graph.get_tensor_by_name(op_name) for op_name in self.activation_names]

    @property
    def data_ops(self):
        # type: () -> List[tf.Tensor, ...]
        """Ops to evaluate and store at each run"""
        return self.activation_ops

    @data_ops.setter
    def data_ops(self, ops):
        # type: (Tuple[np.ndarray, ...]) -> None
        self.activation_acc.append(ops)

    def collect_batch(self, batch):
        # type: (Tuple[Any, ...]) -> None
        """Collect data used in eval"""
        self.inputs_acc.append(batch)

    def collect_vars(self, sess):
        # type: (tf.Session) -> None
        """Save trained variables"""
        import os
        from .layers import NciKeys
        file_name = os.path.join(self.archive_dir, 'vars.npz')

        scope = []
        for key in [NciKeys.INF_VARIABLES, NciKeys.AE_DECODER_VARIABLES, NciKeys.AE_ENCODER_VARIABLES]:
            scope += sess.graph.get_collection(key)

        trained_vars = {var.op.name: var.eval(sess) for var in scope}

        with open(file_name, 'w') as vars_file:
            np.savez(vars_file, **trained_vars)

    def save(self):
        """Save the data collected throughout evaluation"""
        import os
        file_name = os.path.join(self.archive_dir, self.archive_name + '.npz')

        # assemble the collected data
        names = ['results', 'names', 'fingerprints', 'labels']
        names.extend(self.activation_names)

        input_data = map(np.concatenate, zip(*self.inputs_acc))
        act_data = map(np.concatenate, zip(*self.activation_acc))

        results = dict(zip(names, list(input_data) + list(act_data)))

        # save the file
        with open(file_name, 'wb') as result_file:
            np.savez(result_file, **results)


class TopoSessionConfig(InfSessionConfig):
    """Network which learns protein topology from the autoencoder latent space."""
    inf_type = 'topo'
    model_config = InfConfig(label_type='topologies')

    def batch_gen(self):
        """Supplies batches of NCI fingerprints and topologies."""
        return training_inputs(eval_data=False, batch_size=self.train_config.batch_size,
                               request=self.request, ingest_config=self.ingest_config,
                               data_types=('fingerprints', 'topologies'))


class SignSessionConfig(InfSessionConfig):
    """Learns whether the stability score is > 0 from autoencoder latent space."""
    inf_type = 'sign'
    model_config = InfConfig(n_logits=2, label_type='scores')

    def batch_gen(self):
        """Supplies batches of NCI fingerprints and stability scores."""
        return training_inputs(eval_data=False, batch_size=self.train_config.batch_size,
                               request=self.request, ingest_config=self.ingest_config,
                               data_types=('fingerprints', 'scores'))
