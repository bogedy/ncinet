"""
Base classes for config data structures.
"""

from __future__ import division, print_function

import tensorflow as tf
import numpy as np

from typing import Any, Iterator, Mapping, Tuple


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
    pass


@freeze
class TrainingConfig(ConfigBase):
    """Configuration for training scheme."""
    batch_size = None                       # type: int
    train_dir = None                        # type: str

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
class SessionConfig(ConfigBase):
    """Main container for all model configurations."""
    xent_type = None                        # type: str
    model_config = None                     # type: ModelConfig
    train_config = None                     # type: TrainingConfig

    @property
    def batch_gen_args(self):
        # type: () -> Mapping[str, Any]
        """Dict of arguments to pass to `nci_input.inputs`"""
        raise NotImplemented

    def logits_network_gen(self, graph, config):
        # type: (tf.Graph, ModelConfig) -> Tuple[tf.Tensor, tf.Tensor]
        """Constructs main network. Returns logits, labels tensors"""
        raise NotImplemented

    def batch_gen(self):
        # type: () -> Iterator[Tuple[tf.Tensor, tf.Tensor]]
        """Loads data for training.

        Calls nci_input.inputs and processes the result to return a print
        batch and a label batch as needed for the particular model to be
        trained. Also updates `num_examples_per_epoch_train` in `train_config`.
        """
        raise NotImplemented
