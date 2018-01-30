"""
Classes to hold configurations for different parts of the model
"""

import tensorflow as tf

from typing import Any, Iterator, Mapping, Callable, Tuple


def freeze(cls):
    cls.__frozen = False

    def frozen_setattr(self, key, value):
        """Prevents creation of attributes on frozen classes"""
        if self.__frozen and not hasattr(self, key):
            raise ValueError("Unknown attribute " + key)
        else:
            object.__setattr__(self, key, value)

    def init_decorator(func):
        from functools import wraps
        @wraps(func)
        def wrapper(self, *args,  **kwargs):
            func(self, *args, **kwargs)
            self.__frozen = True

        return wrapper

    cls.__setattr__ = frozen_setattr
    cls.__init__ = init_decorator(cls.__init__)

    return cls


@freeze
class ModelConfig(object):
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class EncoderConfig(ModelConfig):
    """Configuration for half of an autoencoder.

    Attributes frozen via inherited methods. Parameter lists go from farthest
    from to closest to the latent space. This ordering allows the same class
    to be used to specify both the encoder and the decoder.

    Attributes
    ----------
    n_layers: Int
    n_filters: List[Int]
    filter_size: List[Int]
    reg_weight: List[Float, None]
    init_dim: List[Int]
        Dimension of the side of the layer farthest from the latent space.
    """
    def __init__(self):
        self.n_layers = 3
        self.n_filters = [32, 32, 16]
        self.filter_size = [5, 5, 5]
        self.reg_weight = [0.001, 0.001, 0.001]
        self.init_dim = [100, 50, 25]
        ModelConfig.__init__(self)


class InfConfig(ModelConfig):
    def __init__(self, **kwargs):
        self.n_hidden = 1
        self.dim_hidden = [128]
        self.n_logits = 4
        self.encoder_config = EncoderConfig()      # type: EncoderConfig
        ModelConfig.__init__(self, **kwargs)


class SessionConfig:
    def logits_network_gen(self, graph, config):
        # type: (tf.Graph, ModelConfig) -> Tuple[tf.Tensor, tf.Tensor]
        raise NotImplemented

    def batch_gen(self):
        # type: () -> Iterator[Tuple[tf.Tensor, tf.Tensor]]
        raise NotImplemented

    batch_gen_args = {}             # type: Mapping[str, Any]
    xent_type = ''                  # type: str
    model_config = None             # type: ModelConfig
