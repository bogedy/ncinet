"""
Loads data from a premade archive. Handles train/test splits,
cross validation generation, and creation of the final
batch generator.
"""

from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np

from .model import WORK_DIR
from .data_ingest import load_data_from_raws


class config:
    archive_dir = WORK_DIR
    full_archive_name = "data_full.npz"
    archive_prefix = "data"
    tt_tags = ("train", "eval")
    n_folds = 5
    xv_tags = ("xvTrain", "xvVal")


def load_data_from_archive(archive_path):
    """Load a dataframe from a processed archive."""

    # Extract archive data into a dict
    with np.load(archive_path) as loaded_data:
        data = {key: loaded_data[key] for key in loaded_data.files}

    return data


# TODO: consider local normalization
def normalize_prints(eval_batch, train_batch):
    """Performs a min-max norm per-pixel based on training data"""
    min_a, max_a = train_batch.min(axis=0), train_batch.max(axis=0)
    dif = max_a - min_a
    dif[dif == 0.] = 1.

    return (eval_batch - min_a) / dif, (train_batch - min_a) / dif


def split_train_eval(fraction=0.1, topo=None):
    """Makes a train-test split of full data archive"""

    # Load data from main archive
    archive_path = os.path.join(config.archive_dir, config.full_archive_name)
    if not os.path.exists(archive_path):
        load_data_from_raws()

    data_dict = load_data_from_archive(archive_path)

    # select for topology
    if topo is not None:
        mask = (data_dict['topologies'] == topo)
        for k in data_dict:
            data_dict[k] = data_dict[k][mask]

    n_tot = next(iter(data_dict.values())).shape[0]
    assert False not in [a.shape[0] == n_tot for a in data_dict.values()]
    n_eval = int(np.floor(n_tot * fraction))

    # split into eval and train data
    idx_shuffle = np.random.permutation(np.arange(n_tot))
    eval_data = {}
    train_data = {}

    for k in data_dict:
        eval_data[k] = data_dict[k][idx_shuffle[:n_eval]]
        train_data[k] = data_dict[k][idx_shuffle[n_eval:]]

    # normalize the fingerprints
    eval_data['fingerprints'], train_data['fingerprints'] = \
        normalize_prints(eval_data['fingerprints'], train_data['fingerprints'])

    # save arrays
    t_str = "" if topo is None else "_t{}".format(topo)
    name_fstring = "{prefix}_{{batch}}{t}.npz".format(prefix=config.archive_prefix, t=t_str)

    np.savez(os.path.join(config.archive_dir, name_fstring.format(batch=config.tt_tags[0])), **train_data)
    np.savez(os.path.join(config.archive_dir, name_fstring.format(batch=config.tt_tags[1])), **eval_data)


def split_xval(archive_path, folds=5, fraction=0.2):
    """Splits an archive in several ways for cross validation"""
    import re

    # Get the archive base
    basename = os.path.basename(archive_path)
    pattern = "([\w]+)_({tags})(_t([\d]{2}))?"
    match = re.match(pattern, basename)
    if match:
        prefix = match.group(1)
        topo = match.group(3)
        topo = topo if topo is not None else ''
    else:
        raise ValueError

    # Make a list of archives to create
    name_base = prefix + "_{}{:02}" + topo + ".npz"
    train_archives = [name_base.format(config.xv_tags[0], i+1) for i in range(folds)]
    val_archives = [name_base.format(config.xv_tags[1], i+1) for i in range(folds)]

    # Load the data to split
    full_data = load_data_from_archive(archive_path)

    # Calculate split size
    n_tot = next(iter(full_data.values())).shape[0]
    assert False not in [a.shape[0] == n_tot for a in full_data.values()]
    n_eval = int(np.floor(n_tot * fraction))

    for t_name, v_name in zip(train_archives, val_archives):
        # Delete any existing archive
        if os.path.exists(t_name):
            os.remove(t_name)
        if os.path.exists(v_name):
            os.remove(v_name)

        # split into eval and train data
        shuffle = np.random.permutation(np.arange(n_tot))
        t_data = {}
        v_data = {}

        for k in full_data:
            v_data[k] = full_data[k][shuffle[:n_eval]]
            t_data[k] = full_data[k][shuffle[n_eval:]]

        # Save data
        np.savez(t_name, **t_data)
        np.savez(v_name, **v_data)


def load_data(eval_data, topo=None, fold=None, new_split=False, reload_data=False):
    """Load necessary data archive. Regenerate archives as needed.

    Generates names based on the global `config` class. Checks for the
    presence of the requested archive in `config.archive_dir`. Expects
    archive names to be formatted as
        PREFIX_TAG[_TOPO].npz

    Parameters
    ----------
    eval_data: Bool
        Whether to use training or eval data. If true, uses eval data.
        In tag lists, assumes the ordering (train, eval).
    topo: None | Int
        If given, splits are made using only protiens of a given topology.
    fold: None | Int
        If given, selects the Nth train/validation split for cross-validation.
    new_split: Bool
        Make a new train-test split. Also makes a new xval split if
        `fold` is not None. Default False.
    reload_data: Bool
        Reloads data from raw fingerprint files and score tables. Also
        recreates

    Returns
    -------
    Mapping[str, NDArray]
        Data dictionary returned by `load_data_from_archive`
    """
    # Get appropriate paths for full and xv archives
    t_str = '' if topo is None else "_t{}".format(topo)
    name_fstr = config.archive_prefix + "_{tag}" + t_str + ".npz"
    tt_names = [name_fstr.format(tag=tag) for tag in config.tt_tags]
    xv_names = [] if topo is None else [name_fstr.format(tag=tag) for tag in config.xv_tags]

    # Decide which archives to regenerate
    new_tt_split = reload_data or new_split or \
        (False in [os.path.exists(os.path.join(config.archive_dir, f_name)) for f_name in tt_names])
    new_xv_split = new_tt_split or \
        (False in [os.path.exists(os.path.join(config.archive_dir, f_name)) for f_name in xv_names])

    # Regenerate source archives if needed
    if reload_data:
        load_data_from_raws()
    if new_tt_split:
        split_train_eval(topo=topo)
    if new_xv_split:
        base_archive = tt_names[0] if not eval_data else tt_names[1]
        split_xval(base_archive, folds=config.n_folds, fraction=(1/config.n_folds))

    # Load data from selected archive
    a_names = tt_names if fold is None else xv_names
    a_idx = 1 if eval_data else 0
    data_file = a_names[a_idx]
    data_dict = load_data_from_archive(data_file)
    n_tot = next(iter(data_dict.values())).shape[0]
    assert False not in [a.shape[0] == n_tot for a in data_dict.values()]

    print("imported {} datapoints".format(n_tot))
    return data_dict


def inf_datagen(arrays, batch, repeat=True):
    """Creates a generator to supply training/eval data

    Parameters
    ----------
    arrays: List[NDArray]
        All must be the same length
    batch: Int
        Size of arrays in generated list.
    repeat: Bool
        If True, all batches will be of size `batch` and data will be
        re-shuffled when data is exhausted. Otherwise the generator is
        exhausted after one pass through the data. The last batch may
        be smaller than others.

    Yields
    ------
    List[NDArray]
        The list will contain the same number of arrays as `arrays`. The
        kth array contains `batch` elements from the kth element of `arrays`
    """

    n_ell = arrays[0].shape[0]
    assert False not in [a.shape[0] == n_ell for a in arrays]

    if not repeat:
        n_batch = int(math.ceil(n_ell / batch))
        for i in range(n_batch):
            if (i+1) * batch <= n_ell:
                s, e = i*batch, (i+1)*batch
                yield tuple(a[s:e] for a in arrays)
            else:
                assert i*batch < n_ell
                yield tuple(a[i*batch:] for a in arrays)
        return

    idx = 0
    while True:
        if idx + batch < n_ell:
            yield tuple(a[idx:idx+batch] for a in arrays)
            idx += batch
        else:
            print("reshuffling queue")
            idx = 0
            shuffle = np.random.permutation(np.arange(n_ell))
            for i in range(len(arrays)):
                arrays[i] = arrays[i][shuffle]


class DataStream:
    """Wraps a generator to provide length"""
    def __init__(self, length, generator):
        self._length = length
        self._gen = generator

    def __len__(self):
        return self._length

    def __next__(self):
        return next(self._gen)

    def next(self):
        """For python2 compatibility."""
        return self.__next__()


# TODO: refactor to use Dataset objects
# TODO: streamline parameters on this fn
def inputs(eval_data, batch_size, data_types=('fingerprints', 'topologies'), repeat=True, topo=None):
    # mapping of data
    data_dict = load_data(eval_data, topo=topo)
    data_arrs = [data_dict[k] for k in data_types if k in data_dict]
    data_len = len(data_arrs[0])
    data_gen = inf_datagen(data_arrs, batch_size, repeat)

    return DataStream(data_len, data_gen)
