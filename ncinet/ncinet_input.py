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

from typing import Mapping, MutableSequence, List, Tuple, Sized, Iterable

from .data_ingest import load_data_from_raws
from .config_meta import DataIngestConfig, PredictIngestConfig, DataRequest


def load_data_from_archive(archive_path):
    """Load a data dict from a processed archive."""
    # Extract archive data into a dict
    with np.load(archive_path) as loaded_data:
        data = {key: loaded_data[key] for key in loaded_data.files}

    return data


def normalize_prints(eval_batch, train_batch):
    """Performs a min-max norm per-pixel based on training data"""
    min_a, max_a = train_batch.min(axis=0), train_batch.max(axis=0)
    dif = max_a - min_a
    dif[dif == 0.] = 1.

    return (eval_batch - min_a) / dif, (train_batch - min_a) / dif


def split_train_eval(config, fraction=0.1, topo=None):
    # type: (DataIngestConfig, float, int) -> None
    """Makes a train-test split of full data archive"""

    # Load data from main archive
    archive_path = os.path.join(config.archive_dir, config.full_archive_name)
    if not os.path.exists(archive_path):
        load_data_from_raws(config)

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


def stratified_data_split(full_data, fraction, by='topologies'):
    # type: (Mapping[str, np.ndarray], float, str) -> Tuple[Mapping[str, np.ndarray], ...]
    """Splits a data dict, stratified according to the `by` dataset"""
    # get basic info about the data
    n_total = next(iter(full_data.values())).shape[0]
    classes = np.unique(full_data[by])

    # lists to store the shuffles
    v_shuffles = []
    t_shuffles = []

    # construct shuffles for each class
    for c in classes:
        c_idx = np.arange(n_total)[full_data[by] == c]
        c_num = c_idx.shape[0]
        c_num_v = int(np.floor(fraction * c_num))
        c_shuffle = np.random.permutation(c_idx)
        v_shuffles.append(c_shuffle[:c_num_v])
        t_shuffles.append(c_shuffle[c_num_v:])

    # join shuffles
    idx_v = np.random.permutation(np.concatenate(v_shuffles))
    idx_t = np.random.permutation(np.concatenate(t_shuffles))

    # split the data
    v_data = {}
    t_data = {}

    for k in full_data:
        v_data[k] = full_data[k][idx_v]
        t_data[k] = full_data[k][idx_t]

    return v_data, t_data


def split_xval(archive_path, config, folds=5, fraction=0.2, stratify=False):
    # type: (str, DataIngestConfig, int, float, bool) -> None
    """Splits an archive in several ways for cross validation"""
    import re

    # Get the archive base
    root_path, basename = os.path.split(archive_path)
    pattern = "([A-Za-z0-9]+)_([A-Za-z0-9]+)(_t([\d]{2}))?"
    match = re.match(pattern, basename)
    if match:
        prefix = match.group(1)
        topo = match.group(3)
        topo = topo if topo is not None else ''
    else:
        raise ValueError

    # Make a list of archives to create
    name_base = os.path.join(root_path, prefix + "_{}{:02}" + topo + ".npz")
    train_archives = [name_base.format(config.xv_tags[0], i) for i in range(folds)]
    val_archives = [name_base.format(config.xv_tags[1], i) for i in range(folds)]

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
        if stratify:
            v_data, t_data = stratified_data_split(full_data, fraction)
        else:
            shuffle = np.random.permutation(np.arange(n_tot))
            t_data = {}
            v_data = {}

            for k in full_data:
                v_data[k] = full_data[k][shuffle[:n_eval]]
                t_data[k] = full_data[k][shuffle[n_eval:]]

        # Save data
        np.savez(t_name, **t_data)
        np.savez(v_name, **v_data)


def load_data(eval_data, request, ingest_config, new_split=False, reload_data=False):
    # type: (bool, DataRequest, DataIngestConfig, bool, bool) -> Mapping[str, np.ndarray]
    """Load necessary data archive. Regenerate archives as needed.

    Generates names based on the `ingest_config` parameter. Checks for the
    presence of the requested archive in `ingest_config.archive_dir`. Expects
    archive names to be formatted as
        PREFIX_TAG[_TOPO].npz

    Parameters
    ----------
    eval_data: Bool
        Whether to use training or eval data. If true, uses eval data.
        In tag lists, assumes the ordering (train, eval).
    request: DataRequest
        parameters for
    ingest_config: DataIngestConfig
        configuration passed to archive constructors
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
    t_str = '' if request.topo_restrict is None else "_t{}".format(request.topo_restrict)
    name_fstr = ingest_config.archive_prefix + "_{tag}" + t_str + ".npz"
    tt_names = [name_fstr.format(tag=tag) for tag in ingest_config.tt_tags]
    xv_names = [] if request.fold is None else\
        [name_fstr.format(tag=tag+"{:02}".format(request.fold)) for tag in ingest_config.xv_tags]

    # Decide which archives to regenerate
    new_tt_split = reload_data or new_split or \
        (False in [os.path.exists(os.path.join(ingest_config.archive_dir, f_name)) for f_name in tt_names])
    new_xv_split = new_tt_split or \
        (False in [os.path.exists(os.path.join(ingest_config.archive_dir, f_name)) for f_name in xv_names])

    # Regenerate source archives if needed
    if reload_data:
        load_data_from_raws(ingest_config)
    if new_tt_split:
        split_train_eval(ingest_config, topo=request.topo_restrict)
    if new_xv_split and (request.fold is not None):
        base_archive = tt_names[0] if not eval_data else tt_names[1]
        split_xval(os.path.join(ingest_config.archive_dir, base_archive), ingest_config,
                   folds=request.n_folds, fraction=(1 / request.n_folds), stratify=request.stratify)

    # Load data from selected archive
    arch_names = tt_names if request.fold is None else xv_names
    arch_idx = 1 if eval_data else 0
    data_file = os.path.join(ingest_config.archive_dir, arch_names[arch_idx])
    data_dict = load_data_from_archive(data_file)
    n_tot = next(iter(data_dict.values())).shape[0]
    assert False not in [a.shape[0] == n_tot for a in data_dict.values()]

    print("imported {} datapoints".format(n_tot))
    return data_dict


def inf_datagen(arrays, batch, repeat=True):
    # type: (MutableSequence[np.ndarray], int, bool) -> Iterable[List[np.ndarray]]
    """Creates a generator to supply training/eval data

    Parameters
    ----------
    arrays: Sequence[NDArray]
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


class DataStream(Iterable, Sized):
    """Wraps a generator to provide length"""
    def __init__(self, length, generator):
        self._length = length
        self._gen = generator

    def __len__(self):
        return self._length

    def __next__(self):
        return next(self._gen)

    def __iter__(self):
        return self

    def next(self):
        """For python2 compatibility."""
        return self.__next__()


# TODO: refactor to use Dataset objects
def training_inputs(eval_data, batch_size, request, ingest_config, data_types=('fingerprints', 'topologies'), repeat=True):
    # type: (bool, int, DataRequest, DataIngestConfig, Tuple[str, ...]) -> DataStream
    """Constructs generators for batches of input data"""
    # mapping of data
    data_dict = load_data(eval_data, request, ingest_config)
    data_arrs = [data_dict[k] for k in data_types if k in data_dict]
    data_len = len(data_arrs[0])
    data_gen = inf_datagen(data_arrs, batch_size, repeat)

    return DataStream(data_len, data_gen)


def predict_inputs(ingest_config):
    # type: (PredictIngestConfig) -> DataStream
    """Constructs a batch generator for unlabeled inputs."""
    from ncinet.data_ingest import load_prediction_data

    # Load data from source
    archive_path = os.path.join(ingest_config.archive_dir, ingest_config.archive_name)
    if not os.path.exists(archive_path):
        load_prediction_data(ingest_config)

    data_dict = load_data_from_archive(archive_path)

    # Construct data generator
    input_arrs = [data_dict['names'], data_dict['fingerprints']]
    data_len = len(input_arrs[0])
    data_gen = inf_datagen(input_arrs, batch=ingest_config.batch_size, repeat=False)

    return DataStream(data_len, data_gen)
