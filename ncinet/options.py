"""
Defines the ncinet option parser.
"""

from argparse import ArgumentParser
from . import __version__


def parse_args():
    version = "%(prog)s {}".format(__version__)

    parser = ArgumentParser(prog="ncinet")
    parser.add_argument('--version', action='version', version=version)

    # whether to train or evaluate the model
    train_grp = parser.add_mutually_exclusive_group(required=True)
    train_grp.add_argument('--train', action='store_true', dest='train',
                           help="Train the model")
    train_grp.add_argument('--eval', action='store_false', dest='train',
                           help="Evaluate the latest checkpoint")

    # which model to train
    model_grp_w = parser.add_argument_group(title="Model type options")
    model_grp = model_grp_w.add_mutually_exclusive_group()
    model_grp.add_argument('--ae', action='store_const', dest='model', const='AE',
                           help="Build autoencoder model")
    model_grp.add_argument('--topo', action='store_const', dest='model', const='topo',
                           help="Build inference model for topology prediction")
    model_grp.add_argument('--sign', action='store_const', dest='model', const='sign',
                           help="Build inference model for predicting the sign of stability")

    # TODO: this doesn't play well with mutually exclusive train/eval
    parser.add_argument('--grid', action='store', dest='grid',
                        help="Run hyperparameter optimization")

    parser.add_argument('--out_file', action='store', dest='output',
                        help="file to write optimization")

    # specify work directory
    parser.add_argument('--work_dir', action='store', type=str,
                        help="Directory to write model outputs")

    parser.add_argument('--t_rest', action='store', type=int, dest='topo_restrict', default=None,
                        help="Train model on topology T", metavar="T")

    # Training details (max steps, time, learning params, hyper params, checkpoint interval, ...)

    # Eval details

    # things to do with data to loading

    # config file

    args = parser.parse_args()
    return args
