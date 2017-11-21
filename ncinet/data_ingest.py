"""
Ingest data and store it in numpy arrays.
"""

import os

import numpy as np

from ncinet_input import WORK_DIR

FINGERPRINT_DIR = '/work/projects/SD2E-Community/prod/data/shared-q0-hackathon/team10/nci_rocklin'
SCORE_PATH = os.path.join(WORK_DIR, 'output.csv')


def get_fingerprint_filenames(directory):
    """Create a list of all fingerprint files in a directory.
    Args:
        directory: Directory to be searched.
    Returns:
        List of absolute file names of discovered fingerprint files.
    """
    import re

    f_name_pattern = '(HHH|EHEE|HEEH|EEHEE)_rd[1-4]_[\d]{4}-2d\.dat'
    pattern = re.compile(f_name_pattern)

    def name_match(pattern, name):
        """Asserts that the regexp matches the entire name"""
        match = pattern.match(name)
        if match:
            return match.end() == len(name)
        else:
            return False

    fingerprint_paths = [os.path.join(directory, f)
                         for f in os.listdir(directory)
                         if name_match(pattern, f)]

    return fingerprint_paths


def get_design_name(f_name):
    """Extract the design name from a file name."""
    import re

    f_name = os.path.basename(f_name)
    design_pattern = '(HHH|EHEE|HEEH|EEHEE)_rd[1-4]_[\d]{4}'
    match = re.match(design_pattern, f_name)

    if match:
        return f_name[match.start():match.end()]
    else:
        raise ValueError


# load stability scores from CSV
# returns a dict of name:score pairs and a list names not in CSV
def load_stab_scores(score_path):
    import csv

    with open(score_path, 'r') as score_file:
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


def extract_topology(f_name):
    """Infers topology from filename and returns a code in [0,3]"""
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
    return topology


def load_fingerprint(f_name):
    """Loads a fingerprint from a dat file.
    Returns a [n, n, 1] array"""
    # TODO: remove hardcoded 100x100 shape
    return np.flipud(np.loadtxt(f_name,np.float32).reshape(100, 100).T).reshape(100, 100, 1)


# reloads data from CSV and integration files
# saves to a numpy archive
def load_data_from_raws(work_dir=WORK_DIR):
    import shutil

    print("reloading raw data")
    work_dir = os.path.abspath(work_dir)

    # remove work directory (to clear any old sessions, etc)
    try:
        shutil.rmtree(work_dir)
    except FileNotFoundError:
        pass

    # regenerate work dir
    os.makedirs(work_dir)

    fingerprint_names = get_fingerprint_filenames(FINGERPRINT_DIR)
    stability_table, no_data = load_stab_scores(SCORE_PATH)

    design_names = []
    fingerprints = []
    stab_scores = []
    topos = []

    for f_name in fingerprint_names:
        if f_name not in no_data:
            design_name = get_design_name(f_name)
            score = get_stab_score(stability_table, design_name)
            fingerprint_arr = load_fingerprint(f_name)
            topology = extract_topology(f_name)

            if score is not None:
                design_names.append(design_name)
                fingerprints.append(fingerprint_arr)
                stab_scores.append(score)
                topos.append(topology)
        else:
            print("no data for " + f_name)

    # TODO: string sizing
    names_all = np.array(design_names, dtype='U15')
    prints_all = np.array(fingerprints, dtype=np.float32)
    scores_all = np.array(stab_scores, dtype=np.float32)
    topos_all = np.array(topos, dtype=np.int32)

    # save full files
    np.savez(os.path.join(work_dir, "data_full.npz"),
             **{'names': names_all, 'fingerprints': prints_all, 'scores': scores_all, 'topologies': topos_all})

    print("raw data loaded and saved")
