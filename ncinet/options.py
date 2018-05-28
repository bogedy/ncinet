"""
Defines the ncinet option parser.
"""

from argparse import ArgumentParser, Action
from . import __version__


def FlagAction(flag_key):
    class ModeAction(Action):
        """Setting multiple flags in options namespace"""
        def __call__(self, parser, namespace, values, options_string=None):
            # Set the mode selector
            setattr(namespace, flag_key, self.dest)

            # Set the mode specific args
            setattr(namespace, self.dest, values)
    return ModeAction


def parse_args():
    """Parse ncinet args off of argv"""
    version = "%(prog)s {}".format(__version__)

    arg_parser = ArgumentParser(prog="ncinet")
    arg_parser.add_argument('--version', action='version', version=version)

    # Evaluation mode
    mode_grp = arg_parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument('--train', action='store_const', const='train', dest='mode',
                          help="Train the model")
    mode_grp.add_argument('--eval', action='store_const', const='eval', dest='mode',
                          help="Evaluate the latest checkpoint")
    mode_grp.add_argument('--xval', action='store_const', const='xval', dest='mode',
                          help="Evaluate the latest checkpoint")
    mode_grp.add_argument('--serialize', action='store_const', const='serialize', dest='mode',
                          help="Serialize a trained model and write to disk")

    mode_action = FlagAction('mode')
    mode_grp.add_argument('--predict', action=mode_action, nargs=2, metavar=('MODEL_PATH', 'DATA_CONFIG'),
                          help="Predict new results from a trained model.")

    mode_grp.add_argument('--grid', action=mode_action,
                          metavar='GRID_CONF', help="Run grid search hyperparameter optimization")

    mode_grp.add_argument('--rand', action=mode_action, nargs=2, metavar=('RAND_CONF', 'N_RUNS'),
                          help="Optimize parameters through random selection")

    # which model to train
    model_grp_w = arg_parser.add_argument_group(title="Model type options")
    model_grp = model_grp_w.add_mutually_exclusive_group()
    model_grp.add_argument('--ae', action='store_const', dest='model', const='encoder',
                           help="Build autoencoder model")
    model_grp.add_argument('--topo', action='store_const', dest='model', const='topo',
                           help="Build inference model for topology prediction")
    model_grp.add_argument('--sign', action='store_const', dest='model', const='sign',
                           help="Build inference model for predicting the sign of stability")
    model_grp.add_argument('--stable', action='store_const', dest='model', const='stable',
                           help="Build a model to predict the `stable?` column.")
    model_grp.add_argument('--conf', action=FlagAction('model'), metavar='CONFIG',
                           help="Specify hyperparameters via a config file.")

    arg_parser.add_argument('--out_file', action='store', dest='output',
                            help="File to write optimization")

    # specify work directory
    arg_parser.add_argument('--work_dir', action='store', type=str,
                            help="Directory to write model outputs")

    arg_parser.add_argument('--basename', action='store', type=str,
                            help="Part of naming of work directory.")

    arg_parser.add_argument('--t_rest', action='store', type=int, dest='topo_restrict', default=None,
                            help="Train model on topology T", metavar="TOPO")

    args = arg_parser.parse_args()
    return args
