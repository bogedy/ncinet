"""
Base machinery for automatically creating config objects.
"""
import os

import numpy as np


class Parameter:
    """Parameter placeholder for use in random search"""
    def __init__(self, dist=None, values=None):
        self._iter = None
        self.dist = dist
        self.values = values

    def render(self):
        """Pick a random parameter"""
        if self.dist is not None:
            return self.dist.rvs()
        else:
            return np.random.choice(self.values)

    def __iter__(self):
        if self._iter is None:
            self._iter = self.values.__iter__()
        return self._iter

    def __next__(self):
        return next(self.__iter__())

    def __repr__(self):
        return "{name}(dist={dist}, values={values})".format(
            name=self.__class__.__name__, dist=repr(self.dist), values=self.values)


class ParamTuple:
    """Tuple of parameter objects"""
    def __init__(self, base):
        self.base = base

    def render(self):
        """Generate random parameter tuple."""
        return tuple(map(lambda x: x.render(), self.base))

    def __repr__(self):
        return "{name}(base={base})".format(name=self.__class__.__name__, base=repr(self.base))


def dict_product(param_dict):
    """Enumerate all possible conditions for a grid search"""
    from itertools import product
    return (dict(zip(param_dict, x)) for x in product(*param_dict.values()))


def make_config(params, fstring):
    """construct a full session config with given params"""
    from ncinet import WORK_DIR, FINGERPRINT_DIR
    from ncinet.config_hyper import EncoderConfig
    from ncinet.config_init import EncoderSessionConfig
    from ncinet.config_meta import DataIngestConfig, TrainingConfig, EvalConfig, EvalWriterBase

    # make base name
    conditions = fstring(**params)
    train_dir = os.path.join(WORK_DIR, "{}_train".format(conditions))
    eval_dir = os.path.join(WORK_DIR, "{}_eval".format(conditions))

    ingest_config = DataIngestConfig(archive_dir=WORK_DIR,
                                     fingerprint_dir=FINGERPRINT_DIR,
                                     score_path=os.path.join(WORK_DIR, "../output.csv"),
                                     archive_prefix="data")

    training_config = TrainingConfig(train_dir=train_dir,
                                     batch_size=params['train_batch_size'],
                                     num_examples_per_epoch_train=14000,
                                     max_steps=params['max_steps'],
                                     initial_learning_rate=params['initial_learning_rate'],
                                     num_epochs_per_decay=params['epochs_per_decay'])

    eval_config = EvalConfig(batch_size=100,
                             eval_dir=eval_dir,
                             train_dir=train_dir,
                             run_once=True,
                             data_writer=EvalWriterBase())

    model_config = EncoderConfig(n_layers=3,
                                 n_filters=params['n_filters'],
                                 filter_size=params['filter_size'],
                                 reg_weight=params['reg_weight'],
                                 init_dim=[100, 50, 25])

    session_config = EncoderSessionConfig(model_config=model_config,
                                          train_config=training_config,
                                          eval_config=eval_config,
                                          ingest_config=ingest_config)
    return session_config


def ae_fstring(n_filters=None, reg_weight=None, initial_learning_rate=None, **kw):
    """Format string for autoencoder"""
    filters_str = ".".join(map(str, n_filters))
    return "AE_{filters}_reg{reg[0]:.1e}_lr{lr:.1e}".format(
        filters=filters_str, reg=reg_weight, lr=initial_learning_rate)
