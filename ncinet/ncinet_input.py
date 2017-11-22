#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import math
import os

import numpy as np

import model
from data_ingest import load_data_from_raws

WORK_DIR = model.WORK_DIR

N_EVAL = 600


def load_data_from_archive():
    """Load full dataframe from processed archives."""
    # TODO: hardcoded param
    archive_path = os.path.join(WORK_DIR, "data_full.npz")
    if not os.path.exists(archive_path):
        load_data_from_raws()

    with np.load(archive_path) as loaded_data:
        names = loaded_data['names']
        prints = loaded_data['fingerprints']
        scores = loaded_data['scores']
        topols = loaded_data['topologies']

    return names, prints, scores, topols


# TODO: consider local normalization
def normalize_prints(eval_batch, train_batch):
    """Performs a min-max norm per-pixel based on training data"""
    min_a, max_a = train_batch.min(axis=0), train_batch.max(axis=0)
    dif = max_a - min_a
    dif[dif == 0.] = 1.

    return (eval_batch - min_a) / dif, (train_batch - min_a) / dif


def split_data(fraction=0.1, n_splits=1, prefix="data"):
    # load data from main source
    names, fingerprints, scores, topols = load_data_from_archive()
    n_tot = scores.shape[0]
    assert names.shape[0] == fingerprints.shape[0] == scores.shape[0] == topols.shape[0]
    n_eval = int(np.floor(n_tot * fraction))

    for i in range(n_splits):
        # split into eval and train data
        idx_shuffle = np.random.permutation(np.arange(n_tot))

        # TODO: there must be a more extensible way to do this
        # shuffle and split the data into train and eval
        eval_names = names[idx_shuffle[:n_eval]]
        eval_print = fingerprints[idx_shuffle[:n_eval]]
        eval_score = scores[idx_shuffle[:n_eval]]
        eval_topol = topols[idx_shuffle[:n_eval]]

        train_names = names[idx_shuffle[n_eval:]]
        train_print = fingerprints[idx_shuffle[n_eval:]]
        train_score = scores[idx_shuffle[n_eval:]]
        train_topol = topols[idx_shuffle[n_eval:]]

        # normalize the fingerprints
        eval_print, train_print = normalize_prints(eval_print, train_print)

        # save arrays
        name_fstring = "{prefix}_{batch}_{num:02}.npz"
        np.savez(os.path.join(WORK_DIR, name_fstring.format(prefix=prefix, batch="eval", num=i)),
                 **{'names': eval_names, 'fingerprints': eval_print, 'scores': eval_score, 'topologies': eval_topol})

        np.savez(os.path.join(WORK_DIR, name_fstring.format(prefix=prefix, batch="train", num=i)),
                 **{'names': train_names, 'fingerprints': train_print, 'scores': train_score, 'topologies': train_topol})


# load training or eval data. Potentially re_split data
def load_data(eval_data, new_split=False, reload_data=False):

    # check archive existence
    data_file = "data_eval_00.npz" if eval_data else "data_train_00.npz"
    data_file = os.path.join(WORK_DIR, data_file)

    if reload_data:
        load_data_from_raws()

    if not os.path.exists(data_file) or new_split or reload_data:
        split_data()

    with np.load(data_file) as loaded_data:
        names = loaded_data['names']
        prints = loaded_data['fingerprints']
        scores = loaded_data['scores']
        topols = loaded_data['topologies']

    assert prints.shape[0] == scores.shape[0]
    print("imported %d datapoints" % prints.shape[0])
    # TODO: use this dict structure throughout the pipeline
    return {'names': names, 'fingerprints': prints,
            'scores': scores, 'topologies': topols}


# TODO: this is ugly AND not fully correct
def inf_datagen(arrays, batch, repeat=True):
    n_ell = arrays[0].shape[0]
    idx = 0
    assert False not in [a.shape[0] == n_ell for a in arrays]

    if not repeat:
        n_batch = int(math.ceil(n_ell / batch))
        for i in range(n_batch):
            if (i+1) * batch <= n_ell:
                s, e = i*batch, (i+1)*batch
                yield tuple(a[s, e] for a in arrays)
            else:
                assert i*batch < n_ell
                yield tuple(a[i*batch:] for a in arrays)
        return

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
def inputs(eval_data, batch_size, data_types=('fingerprints', 'topologies')):
    # mapping of data
    data_dict = load_data(eval_data)
    data_arrs = [data_dict[k] for k in data_types if k in data_dict]
    data_len = len(data_arrs[0])
    data_gen = inf_datagen(data_arrs, batch_size, not eval_data)

    return DataStream(data_len, data_gen)
