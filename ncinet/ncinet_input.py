#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math, os

import model


#FINGERPRINT_DIR = '../results'
FINGERPRINT_DIR = '/work/projects/SD2E-Community/prod/data/shared-q0-hackathon/team10/nci_rocklin'
WORK_DIR = model.WORK_DIR
SCORE_PATH = os.path.join(WORK_DIR, 'output.csv')

N_EVAL = 600


# create a list of all integration files
def input_filenames():
    import re

    PATTERN = '(HHH|EHEE|HEEH|EEHEE)_rd[1-4]_[\d]{4}-2d\.dat'
    pattern = re.compile(PATTERN)

    def name_match(pattern, name):
        match = pattern.match(name)

        if match:
            return match.end() == len(name)
        else:
            return False

    fingerprint_paths = [os.path.join(FINGERPRINT_DIR, f)
                         for f in os.listdir(FINGERPRINT_DIR)
                         if name_match(pattern, f)]

    return fingerprint_paths


# extracts the design name from a file name
def get_design_name(f_name):
    import re
    PATTERN = '(HHH|EHEE|HEEH|EEHEE)_rd[1-4]_[\d]{4}'

    f_name = os.path.basename(f_name)
    match = re.match(PATTERN, f_name)

    if match:
        return f_name[match.start():match.end()]
    else:
        raise ValueError


# load stability scores from CSV
# returns a dict of name:score pairs and a list names not in CSV
def load_stab_scores():
    import csv

    with open(SCORE_PATH, 'r') as score_file:
        reader = csv.DictReader(score_file)

        pairs = []
        no_data = []
        for row in reader:
            try:
                pairs.append((get_design_name(row['basename']), float(row['stabilityscore'])))
            except ValueError:
                if row['stabilityscore'] == '':
                    no_data.append(row['stabilityscore'])
                else:
                    raise

    return dict(pairs), no_data


# gets score from dict
def get_stab_score(score_table, name):
    try:
        return score_table[name]
    except KeyError:
        print(name + " not in table")
        return None


# reloads data from CSV and integration files
# saves to a numpy archive
def load_data_from_raws():
    import numpy as np
    import shutil, errno
    print("reloading raw data")

    # remove work directory (to clear any old sessions, etc)
    try:
        shutil.rmtree(WORK_DIR)
    except OSError as e:
        raise e

    # regenerate work dir
    try:
        os.makedirs(WORK_DIR)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    fingerprint_names = input_filenames()
    stability_table, no_data = load_stab_scores()

    fingerprints = []
    stab_scores = []
    topos = []

    # TODO: remove hardcoded 100x100 shape
    for f_name in fingerprint_names:
        if f_name not in no_data:
            score = get_stab_score(stability_table, get_design_name(f_name))
            fingerprint_arr = np.flipud(np.loadtxt(f_name,np.float32).reshape(100,100).T)

            if "HHH" in f_name:
                topology = 0
            elif "EHEE" in f_name:
                topology = 1
            elif "HEEH" in f_name:
                topology = 2
            elif "EEHEE" in f_name:
                topology = 3
            else:
                raise ValueError("Invalid name {}".format(f_name))

            if score is not None:
                fingerprints.append(fingerprint_arr.reshape(100,100,1))
                stab_scores.append(score)
                topos.append(topology)
        else:
            print("no data for " + f_name)

    prints_all = np.array(fingerprints, dtype=np.float32)
    scores_all = np.array(stab_scores, dtype=np.float32)
    topos_all = np.array(topos, dtype=np.int32)

    # save full files
    np.savez(os.path.join(WORK_DIR, "data_full.npz"),
                 **{'fingerprints': prints_all, 'scores': scores_all, 'topologies': topos_all})

    print("data loaded and saved")


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
