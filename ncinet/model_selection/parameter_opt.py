"""
Evaluates performance via cross validation for hyperparameter optimization.
"""

import os
import numpy as np
from datetime import datetime

import ncinet.eval
import ncinet.train

from ncinet.config_meta import SessionConfig, DataRequest
from ncinet.model_selection.hyper_parameters import make_config, ae_fstring


def add_dir_index(config, fold):
    # type: (SessionConfig, int) -> SessionConfig
    """Updates a config to index train and eval dirs by fold"""
    def update_dir_name(dir_path, fold):
        # type: (str, int) -> str
        """Adds a fold index to a single path"""
        if dir_path is not None:
            head, tail = os.path.split(dir_path)
            tail = tail + str(fold)
            return os.path.join(head, tail)

    config.train_config.train_dir = update_dir_name(config.train_config.train_dir, fold)
    config.eval_config.train_dir = update_dir_name(config.eval_config.train_dir, fold)
    config.eval_config.eval_dir = update_dir_name(config.eval_config.eval_dir, fold)

    return config


def eval_fold(base_config, fold, n_folds):
    """Train and evaluate a model on one fold"""
    from copy import deepcopy
    # copy the config and update training paths
    config = deepcopy(base_config)
    config = add_dir_index(config, fold)

    # set up data request
    request = DataRequest(n_folds=n_folds, fold=fold)
    config.request = request

    # train the model
    print("{}: Training model on split {}".format(datetime.now(), fold))
    ncinet.train.main(config=config)

    # evaluate the model
    print("{}: Evaluating model on split {}".format(datetime.now(), fold))
    eval_result = ncinet.eval.main(config=config)

    return eval_result


def xval_condition(config, n_folds):
    """Cross validates eval metrics for a condition"""
    eval_results = []
    for fold in range(n_folds):
        result = eval_fold(config, fold, n_folds)
        eval_results.append(result)

    # join results
    raw_results = {k: np.array([result[k] for result in eval_results]) for k in eval_results[0]}
    result_stats = {k: (a.mean(), a.std()) for k, a in raw_results.items()}

    return result_stats, raw_results


def grid_search(fixed_params, var_params, n_folds=3):
    """Exhaustively evaluate all combinations of parameters"""
    from .hyper_parameters import dict_product

    # results dicts hold both parameter settings and evaluation metrics
    results = []

    # Searches cartesian product of values in `var_params`
    for param_dict in dict_product(var_params):
        param_dict.update(fixed_params)
        print("{}: Using {}".format(datetime.now(), str(param_dict)))

        # Cross validate parameter selection
        config = make_config(param_dict, ae_fstring)
        stats, raw = xval_condition(config, n_folds)

        # print stats
        for k, v in stats.items():
            print("{}: {:.3} +/- {:.3}".format(k, v[0], v[1]))

        # save results
        param_dict.update(raw)
        results.append(param_dict)

    return results


def random_search(fixed_params, var_params, n_iter, n_folds=3):
    """Test `n_iter` random parameter choices"""

    # Hold both parameter settings and xval results
    results = []

    # Cross validate a random parameter selection
    for _ in range(n_iter):
        # select a set of parameters
        param_dict = {k: v.render() for k, v in var_params.items()}
        param_dict.update(fixed_params)
        print("{}: Using {}".format(datetime.now(), str(param_dict)))

        # Do evaluation
        config = make_config(param_dict, ae_fstring)
        stats, raw = xval_condition(config, n_folds)

        # Print stats
        for k, v in stats.items():
            print("{}: {:.3} +/- {:.3}".format(k, v[0], v[1]))

        # Save results
        param_dict.update(raw)
        results.append(param_dict)

    return results


def main():
    fixed_dict = dict(max_steps=2,
                      train_batch_size=64,
                      initial_learning_rate=0.005,
                      epochs_per_decay=50.,
                      filter_size=[5, 5, 5])

    var_dict = dict(n_filters=[[18, 18, 12], [12, 12, 12]],
                    reg_weight=[[0.001]*3, [0.005]*3])

    result = grid_search(fixed_dict, var_dict, 2)
    print(result)
