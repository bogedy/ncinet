"""
Base classes for config data structures. These data-structures are used
to store model hyperparameters, paths to data, and training details. The
config objects are modified to prevent creation of new attributes. They
are initialized using config files with `make_config`.
"""

import tensorflow as tf
import numpy as np

from typing import Any, Iterator, Union, Tuple, List


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
    label_type = None                       # type: str


@freeze
class DataIngestConfig(ConfigBase):
    """Parameters for data ingest and processing.

    ------------
    full_archive_name: str
        Name for the complete archive of available data. Stored
        in `archive_dir`.
    archive_dir: path
        Directory where archives of training data and data splits
        are stored.
    fingerprint_dir: path
        Directory to find NCI fingerprints.
    score_path: path
        Path to a CSV mapping names to stability scores.
    archive_prefix: str
        Combined with tags to name archives of data splits.
    tt_tags: Tuple[str, str]
        Appended to `archive_prefix` to name archives for train/test
        split. Archives are named `[prefix]_[tag].npz`. Values are
        `(train_tag, test_tag)`.
    xv_tags: Tuple[str, str]
        Similar to `tt_tags` but for naming training and validation
        archives for cross validation. Archive name format is
        `[prefix]_[tag]##.npz` and value format is
        `(train_tag, validation_tag)`.
    """
    # Data for constructing archive paths
    full_archive_name = None                # type: str
    archive_dir = None                      # type: str
    nci_dir = None                          # type: str
    score_path = None                       # type: Union[str, Tuple[str, str]]
    ingest_version = 'rocklin_v1'           # type: str
    topo_index_name = None                  # type: str

    # parameters for programmatically constructing archive names
    archive_prefix = None                   # type: str
    tt_tags = None                          # type: Tuple[str, str]
    xv_tags = None                          # type: Tuple[str, str]


@freeze
class PredictIngestConfig(ConfigBase):
    archive_dir = None                      # type: str
    nci_dir = None                          # type: str
    dataframe_path = None                   # type: str
    topo_index_name = None                  # type: str
    archive_name = None                     # type: str
    batch_size = None                       # type: int


@freeze
class DataRequest(ConfigBase):
    """Parameters needed to construct a batch generator.

    ------------
    fold: int
        Which fold to retrieve.
    n_folds: int
        Total number of folds if a new split must be created.
    stratify: bool
        Whether to stratify by topology when generating folds.
    topo_restrict: int
        Only return records matching given topology.
    """
    fold = None                             # type: int
    n_folds = None                          # type: int
    stratify = False                        # type: bool
    topo_restrict = None                    # type: int


@freeze
class TrainingConfig(ConfigBase):
    """Configuration for training scheme.

    ------------
    batch_size: int
        Number of training examples per batch.
    train_dir: path
        Directory to store checkpoints.
    encoder_dir: path
        Directory containing checkpoints from autoencoder training. Used
        by inference networks to use weights from a pre-trained autoencoder
        to process input. May be unset for autoencoder training.
    max_steps: int
        Number of batches on which to train.
    log_frequency: int
        Number of steps between console updates.
    summary_steps: int
        Number of steps between summary updates.
    checkpoint_secs: int
        Seconds between checkpoint saves.
    use_learning_rate_decay: bool
        Whether to decay the learning rate during training.
    num_examples_per_epoch_train: int
    initial_learning_rate: float
    num_epochs_per_decay: float
    learning_rate_decay_factor: float
        Learning rate is decayed from `initial_learning_rate` by a factor of
        `learning_rate_decay_factor` every `epochs_per_decay` epochs.
    input_noise: float
        Factor for white noise added to inputs when training the autonecoder.
    """
    batch_size = None                       # type: int
    train_dir = None                        # type: str
    encoder_dir = None                      # type: str

    # Training parameters
    max_steps = None                        # type: int
    log_frequency = None                    # type: int
    summary_steps = None                    # type: int
    checkpoint_secs = None                  # type: int

    # Constants for learning rate schedule
    use_learning_rate_decay = True
    num_examples_per_epoch_train = None     # type: int
    initial_learning_rate = None            # type: float
    num_epochs_per_decay = None             # type: float
    learning_rate_decay_factor = (1/np.e)
    input_noise = None                      # type: float


@freeze
class EvalConfig(ConfigBase):
    """Parameters for model evaluation."""
    batch_size = None                       # type: int
    eval_dir = None                         # type: str
    train_dir = None                        # type: str
    data_writer = None                      # type: EvalWriterBase

    use_eval_data = True
    run_once = None                         # type: bool
    eval_interval = None                    # type: int


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
        # type: (tf.Graph, ModelConfig, bool) -> tf.Tensor
        """Constructs main network. Returns a logits tensor."""
        raise NotImplemented

    def labels_network_gen(self, graph, eval_net=False):
        # type: (tf.Graph, bool) -> tf.Tensor
        """Serve the labels for the model."""
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
        trained.
        """
        raise NotImplemented
