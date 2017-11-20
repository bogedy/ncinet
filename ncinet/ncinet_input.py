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


# load full dataframe from processed archives
def load_data_from_archive(reload_data=False):
    archive_path = os.path.join(WORK_DIR, "data_full.npz")
    if reload_data or not os.path.exists(archive_path):
        load_data_from_raws()

    with np.load(archive_path) as loaded_data:
        prints = loaded_data['fingerprints']
        scores = loaded_data['scores']
        topols = loaded_data['topologies']

    return prints, scores, topols


# load training or eval data. Potentially re_split data
def load_data(eval_data, new_split=False, reload_data=False):

    # check archive existance
    data_file = "data_eval.npz" if eval_data else "data_train.npz"
    data_file = os.path.join(WORK_DIR, data_file)

    if not os.path.exists(data_file) or new_split or reload_data:
        # load data from main sources
        fingerprints, scores, topols = load_data_from_archive(reload_data)
        assert fingerprints.shape[0] == scores.shape[0] == topols.shape[0]
        n_tot = scores.shape[0]

        # performs min-max norm on a single bin
        # TODO: consider local response normalization
        def normalize(v):
            min_v, max_v = v.min(), v.max()
            if np.around(max_v-min_v, 5) == 0:
                return v - min_v
            else:
                return (v - min_v) / (max_v - min_v)

        # normalize the data before splitting
        fingerprints = np.apply_along_axis(normalize, 0, fingerprints)

        # then split into eval and train data
        idx_shuffle = np.random.permutation(np.arange(n_tot))

        eval_print = fingerprints[idx_shuffle[:N_EVAL]]
        eval_score = scores[idx_shuffle[:N_EVAL]]
        eval_topol = topols[idx_shuffle[:N_EVAL]]

        train_print = fingerprints[idx_shuffle[N_EVAL:]]
        train_score = scores[idx_shuffle[N_EVAL:]]
        train_topol = topols[idx_shuffle[N_EVAL:]]

        # save arrays
        np.savez(os.path.join(WORK_DIR, "data_eval.npz"),
                 **{'fingerprints': eval_print, 'scores': eval_score, 'topologies': eval_topol})

        np.savez(os.path.join(WORK_DIR, "data_train.npz"),
                 **{'fingerprints': train_print, 'scores': train_score, 'topologies': train_topol})

        print("loaded %d files" % n_tot)


    with np.load(data_file) as loaded_data:
        prints = loaded_data['fingerprints']
        scores = loaded_data['scores']
        topols = loaded_data['topologies']

    assert prints.shape[0] == scores.shape[0]
    print("imported %d datapoints" % prints.shape[0])

    return prints, scores, topols


# TODO: this is ugly AND not fully correct
def inf_datagen(arr1, arr2, batch, repeat=True):
    n_ell = arr1.shape[0]
    idx = 0

    assert n_ell == arr2.shape[0]

    if not repeat:
        n_batch = int(math.ceil(n_ell / batch))
        for i in range(n_batch):
            if (i+1) * batch <= n_ell:
                s, e = i*batch, (i+1)*batch
                yield arr1[s:e], arr2[s:e]
            else:
                assert i*batch < n_ell
                yield arr1[i*batch:], arr2[i*batch:]
        return

    while True:
        if idx + batch < n_ell:
            yield arr1[idx:idx+batch], arr2[idx:idx+batch]
            idx += batch
        else:
            print("reshuffling queue")
            idx = 0
            shuffle = np.random.permutation(np.arange(n_ell))
            arr1 = arr1[shuffle]
            arr2 = arr2[shuffle]



def inputs(eval_data, batch_size, label_type='scores'):
    # arrays of the full dataset
    prints_arr, scores_arr, topols_arr = load_data(eval_data)
    label_arr = scores_arr if label_type == 'scores' else topols_arr

    return inf_datagen(prints_arr, label_arr, batch_size, not eval_data)
