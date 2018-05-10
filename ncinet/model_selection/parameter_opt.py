"""
Evaluates performance via cross validation for hyperparameter optimization.
"""

import os
import numpy as np
from datetime import datetime

from ncinet.eval import main as model_eval
from ncinet.train import main as model_train

from ncinet.config_meta import SessionConfig, DataRequest
from ncinet.model_selection.hyper_parameters import make_config, ae_fstring

from typing import Any, Mapping, Dict, Tuple, Union, TypeVar


# -------------------------------------------
# Functions to manipulate config dictionaries
# -------------------------------------------
def dict_product(param_dict):
    """Enumerate all possible conditions for a grid search"""
    from itertools import product
    return (dict(zip(param_dict, x)) for x in product(*param_dict.values()))


# Types for join_dict
Numeric_T = Union[int, float]
ConfDict_T = TypeVar('ConfDict_T', Dict[str, Any])


def join_dict(main_dict, aux_dict):
    # type: (ConfDict_T, Mapping[Tuple[str], Numeric_T]) -> ConfDict_T
    """Update values in a multilevel dict, keys of aux dict are tuples"""
    from copy import deepcopy
    main_dict = deepcopy(main_dict)
    for k_list, v in aux_dict.items():
        to_update = main_dict
        for k in k_list[:-1]:
            to_update = to_update[k]
        to_update[k_list[-1]] = v
    return main_dict


# --------------------------------------------
# Classes to represent random parameter values
# --------------------------------------------
class Parameter:
    """Parameter placeholder for use in random search"""
    def __init__(self, dist=None, values=None):
        assert (dist is not None) or (values is not None)
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


# -------------------------------
# Helpers to run cross validation
# -------------------------------
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
    model_train(config=config)

    # evaluate the model
    print("{}: Evaluating model on split {}".format(datetime.now(), fold))
    eval_result = model_eval(config=config)

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


# ----------------------------
# Parameter search definitions
# ----------------------------
def grid_search(fixed_params, var_params, n_folds=3):
    """Exhaustively evaluate all combinations of parameters"""

    # results dicts hold both parameter settings and evaluation metrics
    results = []

    # Searches cartesian product of values in `var_params`
    for param_dict in dict_product(var_params):
        param_dict = join_dict(fixed_params, param_dict)
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
        rendered = {k: v.render() for k, v in var_params.items()}
        param_dict = join_dict(fixed_params, rendered)
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
