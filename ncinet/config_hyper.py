"""
Configurations for network hyperparameters.
"""

from .config_meta import ModelConfig


class EncoderConfig(ModelConfig):
    """Configuration for half of an autoencoder.

    Attributes frozen via inherited methods. Parameter lists go from farthest
    from to closest to the latent space. This ordering allows the same class
    to be used to specify both the encoder and the decoder.

    Attributes
    ----------
    n_layers: int
    n_filters: List[int]
    filter_size: List[int]
    reg_weight: List[float, None]
    init_dim: List[int]
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
    """Configuration for inference network on the latent space.

    Attributes
    ----------
    n_hidden: int
    dim_hidden: List[int]
    n_logits: int
    encoder_config: EncoderConfig
    """
    def __init__(self, **kwargs):
        self.n_hidden = 1
        self.dim_hidden = [128]
        self.n_logits = 4
        self.encoder_config = EncoderConfig()      # type: EncoderConfig
        ModelConfig.__init__(self, **kwargs)
