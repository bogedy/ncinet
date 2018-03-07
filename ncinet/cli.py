"""
Main interface for ncinet.
"""

import os

from . import WORK_DIR, FINGERPRINT_DIR
from .options import parse_args
from .config_meta import DataIngestConfig, DataRequest, TrainingConfig, EvalConfig
from .config_init import EncoderSessionConfig, TopoSessionConfig, SignSessionConfig, EvalWriter, EvalWriterBase


def standard_config(options, base_name, run_once=True):
    train_dir = os.path.join(WORK_DIR, "train_" + base_name)
    eval_dir = os.path.join(WORK_DIR, "eval_" + base_name)

    ingest_config = DataIngestConfig(archive_dir=WORK_DIR,
                                     fingerprint_dir=FINGERPRINT_DIR,
                                     score_path=os.path.join(WORK_DIR, "../output.csv"),
                                     archive_prefix="data")

    training_config = TrainingConfig(train_dir=train_dir,
                                     batch_size=32,
                                     num_examples_per_epoch_train=14000,
                                     max_steps=100000,
                                     initial_learning_rate=0.005,
                                     num_epochs_per_decay=50.0)

    if not run_once:
        data_writer = EvalWriterBase()
    else:
        op_name = ("max_pooling2d_3/MaxPool:0",)
        file_name = "eval_results_{}".format('eval', base_name)
        data_writer = EvalWriter(archive_name=file_name,
                                 archive_dir=WORK_DIR,
                                 saved_vars=op_name)

    eval_config = EvalConfig(batch_size=100,
                             eval_dir=eval_dir,
                             train_dir=train_dir,
                             run_once=run_once,
                             data_writer=data_writer)

    request = DataRequest()

    if options.model == 'AE':
        session_config_cls = EncoderSessionConfig
    elif options.model == 'topo':
        session_config_cls = TopoSessionConfig
    elif options.model == 'sign':
        session_config_cls = SignSessionConfig
    else:
        raise ValueError

    config = session_config_cls(train_config=training_config,
                                eval_config=eval_config,
                                ingest_config=ingest_config,
                                request=request)

    return config


def cli():
    options = parse_args()

    if options.grid:
        import yaml
        from .model_selection.parameter_opt import grid_search
        with open(options.grid, 'r') as conf_file:
            params = yaml.safe_load(conf_file)

        results = grid_search(**params)

        with open(options.output, 'w') as out_file:
            out_file.write(yaml.dump(results))
        return

    autoencoder = options.model == 'AE'
    base_name = ("" if autoencoder else "inf_") + options.model.lower()

    config = standard_config(options, base_name, run_once=True)

    if not autoencoder:
        config.train_config.encoder_dir = os.path.join(WORK_DIR, "train_ae")

    if options.train:
        from .train import main
        main(config)
    else:
        from .eval import main
        main(config)
