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


def make_config(params, fstring=None, basename=None):
    """construct a full session config with given params"""
    from ncinet import WORK_DIR, FINGERPRINT_DIR
    from ncinet.config_hyper import EncoderConfig, InfConfig
    from ncinet.config_init import EncoderSessionConfig, TopoSessionConfig, SignSessionConfig, EvalWriter
    from ncinet.config_meta import DataIngestConfig, DataRequest, TrainingConfig, EvalConfig, EvalWriterBase

    # make base name, give precedence to parametrically specified basename, then to provided formatter
    if basename:
        pass
    elif fstring:
        basename = fstring(**params)
    else:
        basename = 'model'

    train_dir = os.path.join(WORK_DIR, "{}_train".format(basename))
    eval_dir = os.path.join(WORK_DIR, "{}_eval".format(basename))

    # Make default parameters
    ingest_config = DataIngestConfig(archive_dir=WORK_DIR,
                                     fingerprint_dir=FINGERPRINT_DIR,
                                     score_path=os.path.join(WORK_DIR, "../output.csv"),
                                     archive_prefix="data")

    request = DataRequest()
    for k, v in params['request_config'].items():
        try:
            setattr(request, k, v)
        except ValueError as err:
            print(str(err))

    # Update config
    for k, v in params['ingest_config'].items():
        try:
            setattr(ingest_config, k, v)
        except ValueError as err:
            print(str(err))

    training_config = TrainingConfig(train_dir=train_dir,
                                     batch_size=64,
                                     num_examples_per_epoch_train=14000,
                                     max_steps=10000,
                                     initial_learning_rate=0.005,
                                     num_epochs_per_decay=50.0)

    for k, v in params['training_config'].items():
        try:
            setattr(training_config, k, v)
        except ValueError as err:
            print(str(err))

    # Set up eval writer
    use_writer = params['eval_config'].pop('use_writer', False)
    writer_config = params['eval_config'].pop('writer_config', None)
    if use_writer:
        writer = EvalWriter
        for k, v in writer_config.items():
            try:
                setattr(writer, k, v)
            except ValueError as err:
                print(str(err))
    else:
        writer = EvalWriterBase()

    eval_config = EvalConfig(batch_size=100,
                             eval_dir=eval_dir,
                             train_dir=train_dir,
                             run_once=True,
                             data_writer=writer)

    for k, v in params['eval_config'].items():
        try:
            setattr(eval_config, k, v)
        except ValueError as err:
            print(str(err))

    # Set model specific parameters
    model_type = params.pop('model_type', None)
    if model_type == 'encoder':
        model_config = EncoderConfig(n_layers=3, init_dim=(100, 50, 25))
        session_cls = EncoderSessionConfig
    elif model_type == 'topo' or model_type == 'sign':
        n_logits = 4 if model_type == 'topo' else 2
        session_cls = TopoSessionConfig if model_type == 'topo' else SignSessionConfig
        model_config = InfConfig(n_logits=n_logits)
    else:
        raise ValueError

    for k, v in params['model_config'].items():
        try:
            setattr(model_config, k, v)
        except ValueError as err:
            print(str(err))

    session_config = session_cls(model_config=model_config,
                                 train_config=training_config,
                                 eval_config=eval_config,
                                 ingest_config=ingest_config,
                                 request=request)
    return session_config


def ae_fstring(**kw):
    """Format string for autoencoder"""
    pars = {}
    for v in kw.values():
        try:
            pars.update(v)
        except ValueError:
            pass
    n_filters = pars.pop('n_filters', None)
    reg_weight = pars.pop('reg_weight', None)
    initial_learning_rate = pars.pop('initial_learning_rate', None)
    filters_str = ".".join(map(str, n_filters))
    return "AE_{filters}_reg{reg[0]:.1e}_lr{lr:.1e}".format(
        filters=filters_str, reg=reg_weight, lr=initial_learning_rate)
