"""
Base machinery for automatically creating config objects.
"""
import os
from typing import Any, Mapping, MutableMapping, Union, Callable
from ncinet.config_meta import SessionConfig


def make_config(params, fstring=None, basename=None):
    # type:(MutableMapping[str, Any], Callable, str) -> SessionConfig
    """construct a full session config with given params"""
    from ncinet import BASE_CONFIG, WORK_DIR
    from ncinet.config_hyper import EncoderConfig, InfConfig
    from ncinet.config_init import EncoderSessionConfig, TopoSessionConfig, SignSessionConfig, StableSessionConfig, EvalWriter
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

    def update_config(config, param_dict):
        """Updates values in a config object based on a dict of the same structure"""
        for k, v in param_dict.items():
            try:
                setattr(config, k, v)
            except ValueError as err:
                print(str(err))

    # Make configs with default parameters
    ingest_config = DataIngestConfig(**BASE_CONFIG['ingest_config'])
    request = DataRequest(**BASE_CONFIG['request_config'])
    training_config = TrainingConfig(train_dir=train_dir,
                                     **BASE_CONFIG['training_config'])

    # Set up eval writer
    use_writer = params['eval_config'].pop('use_writer', False)
    writer_config = params['eval_config'].pop('writer_config', {})
    if use_writer:
        archive_name = writer_config.pop('archive_name', '{}_eval_results'.format(basename))
        archive_dir = writer_config.pop('archive_dir', WORK_DIR)
        saved_vars = writer_config.pop('saved_vars', ())
        writer = EvalWriter(archive_name, archive_dir, saved_vars)

        update_config(writer, writer_config)
    else:
        writer = EvalWriterBase()

    eval_config = EvalConfig(eval_dir=eval_dir,
                             train_dir=train_dir,
                             data_writer=writer,
                             **BASE_CONFIG['eval_config'])

    # Update config objects based on provided dict
    update_config(ingest_config, params['ingest_config'])
    update_config(request, params['request_config'])
    update_config(training_config, params['training_config'])
    update_config(eval_config, params['eval_config'])

    # Set model specific parameters
    model_type = params.pop('model_type', None)

    # Map of type keys to SessionConfig subclasses
    config_map = {'topo': TopoSessionConfig,
                  'sign': SignSessionConfig,
                  'stable': StableSessionConfig}

    if model_type == 'encoder':
        model_config = EncoderConfig(n_layers=3, init_dim=(100, 50, 25))
        session_cls = EncoderSessionConfig
    elif model_type in config_map:
        session_cls = config_map[model_type]
        model_config = session_cls.model_config

        # Set up the encoder config (this shouldn't be necessary but we don't save a graphdef, just variable weights.
        # Therefore, we need to know encoder structure to build the trained encoder)
        encoder_params = params['model_config'].pop('encoder_config', {})
        encoder_config = EncoderConfig(**encoder_params)
        model_config.encoder_config = encoder_config
    else:
        raise ValueError

    # Update ModelConfig parameters from config file
    update_config(model_config, params['model_config'])

    # Build the main session
    session_config = session_cls(model_config=model_config,
                                 train_config=training_config,
                                 eval_config=eval_config,
                                 ingest_config=ingest_config,
                                 request=request)
    return session_config


def ae_fstring(**kw):
    # type: (**Union[str, Mapping[str, Any]]) -> str
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
    return "AE_{filters}_reg{reg[0]:.2e}_lr{lr:.2e}".format(
        filters=filters_str, reg=reg_weight, lr=initial_learning_rate)


def inf_fstring(**kw):
    # type: (**Union[str, Mapping[str, Any]]) -> str
    """Format string for inference networks"""
    pars = {}
    for v in kw.values():
        try:
            pars.update(v)
        except ValueError:
            pass

    model_type = kw['model_type']
    encoder_name = os.path.basename(os.path.dirname(pars.pop('encoder_dir')))
    return "Inf_{i_type}_E{encoder}".format(i_type=model_type, encoder=encoder_name)
