"""
Base classes for config data structures.
"""

from __future__ import division, print_function

import tensorflow as tf
import numpy as np

from typing import Any, Iterator, Tuple, List


def freeze(cls):
    """
    Decorator to freeze classes by preventing creation of new attributes
    after the __init__ method is run.
    """
    cls.__frozen = False

    def frozen_setattr(self, key, value):
        """Prevents creation of attributes on frozen classes"""
        if self.__frozen and not hasattr(self, key):
            raise ValueError("Unknown attribute " + key)
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        """Decorates the init method"""
        from functools import wraps

        @wraps(func)
        def wrapper(self, *args,  **kwargs):
            """Freezes class after init function"""
            func(self, *args, **kwargs)
            self.__frozen = True

        return wrapper

    cls.__setattr__ = frozen_setattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


class ConfigBase(object):
    """Base for all configuration classes"""
    def __init__(self, **kw):
        for k, v in kw.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError("Unknown attribute " + k)


@freeze
class ModelConfig(ConfigBase):
    """Configuration for network structure"""
    is_autoencoder = None                   # type: bool
    pass


@freeze
class DataIngestConfig(ConfigBase):
    """Parameters for data ingest and processing"""
    # Data for constructing archive paths
    full_archive_name = "data_full.npz"
    archive_dir = None                      # type: str
    fingerprint_dir = None                  # type: str
    score_path = None                       # type: str

    # parameters for programmatically constructing archive names
    archive_prefix = None
    tt_tags = ("train", "eval")
    xv_tags = ("xvTrain", "xvVal")


@freeze
class DataRequest(ConfigBase):
    """Parameters needed to construct a batch generator"""
    fold = None                             # type: int
    n_folds = None                          # type: int
    stratify = False                        # type: bool
    topo_restrict = None                    # type: int


@freeze
class TrainingConfig(ConfigBase):
    """Configuration for training scheme."""
    batch_size = None                       # type: int
    train_dir = None                        # type: str
    encoder_dir = None                      # type: str

    # Training parameters
    max_steps = None                        # type: int
    log_frequency = 25
    summary_steps = 100
    checkpoint_secs = 120

    # Constants for learning rate schedule
    use_learning_rate_decay = True
    num_examples_per_epoch_train = None     # type: int
    initial_learning_rate = None            # type: float
    num_epochs_per_decay = None             # type: float
    learning_rate_decay_factor = (1/np.e)


@freeze
class EvalConfig(ConfigBase):
    """Parameters for evaluation"""
    batch_size = None                       # type: int
    eval_dir = None                         # type: str
    train_dir = None                        # type: str
    data_writer = None                      # type: EvalWriterBase

    use_eval_data = True
    run_once = None                         # type: bool
    eval_interval = 120


@freeze
class EvalWriterBase:
    """Writes eval data to file."""
    def __init__(self):
        pass

    def setup(self, sess):
        # type: (tf.Session) -> None
        """Initialize the writer"""
        pass

    @property
    def data_ops(self):
        # type: () -> List[tf.Tensor, ...]
        """Records ops for each batch during evaluation"""
        return []

    @data_ops.setter
    def data_ops(self, ops):
        # type: (Tuple[np.ndarray]) -> None
        pass

    def collect_batch(self, batch):
        # type: (Tuple[Any, ...]) -> None
        """Collect the data used for evaluation"""
        pass

    def save(self):
        """Write stored data out to file"""
        pass


@freeze
class SessionConfig(ConfigBase):
    """Main container for all model configurations."""
    xent_type = None                        # type: str
    model_config = None                     # type: ModelConfig
    train_config = None                     # type: TrainingConfig
    eval_config = None                      # type: EvalConfig
    ingest_config = None                    # type: DataIngestConfig
    request = None                          # type: DataRequest

    def logits_network_gen(self, graph, config, eval_net=False):
        # type: (tf.Graph, ModelConfig, bool) -> Tuple[tf.Tensor, tf.Tensor]
        """Constructs main network. Returns logits, labels tensors"""
        raise NotImplemented

    def eval_metric(self, logits, labels):
        # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
        """Metric to record during testing"""
        raise NotImplemented

    def batch_gen(self):
        # type: () -> Iterator[Tuple[tf.Tensor, tf.Tensor]]
        """Loads data for training.

        Calls nci_input.inputs and processes the result to return a print
        batch and a label batch as needed for the particular model to be
        trained. Also updates `num_examples_per_epoch_train` in `train_config`.
        """
        raise NotImplemented
