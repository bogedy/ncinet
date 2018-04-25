"""
Provide an api for scripting. The intended use is `from ncinet.names import *`
"""

from ncinet.config_meta import DataIngestConfig, DataRequest, TrainingConfig, EvalConfig, EvalWriterBase
from ncinet.config_hyper import EncoderConfig, InfConfig
from ncinet.config_init import EncoderSessionConfig, InfSessionConfig, SignSessionConfig, TopoSessionConfig, EvalWriter

from ncinet.model_selection.hyper_parameters import make_config
from ncinet.cli import standard_config

from ncinet.train import main as train_model
from ncinet.eval import main as eval_model
from ncinet.model_selection.parameter_opt import xval_condition, grid_search, random_search

config_classes = [DataRequest, DataIngestConfig, TrainingConfig, EvalConfig, EvalWriter, EvalWriterBase,
                  EncoderConfig, InfConfig, EncoderSessionConfig, InfSessionConfig, SignSessionConfig,
                  TopoSessionConfig]
config_helpers = [make_config, standard_config]
execution_methods = [train_model, eval_model, xval_condition, grid_search, random_search]

__all__ = config_classes + config_helpers + execution_methods
