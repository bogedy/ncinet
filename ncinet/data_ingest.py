"""
Ingest data and store it in numpy arrays.
"""

import os
import numpy as np

from . import data_config as config


def get_fingerprint_filenames(directory):
    """Create a list of all fingerprint files in a directory.
    Args:
        directory: Directory to be searched.
    Returns:
        List of absolute file names of discovered fingerprint files.
    """
    import re

    f_name_pattern = '(HHH|EHEE|HEEH|EEHEE)_rd[1-4]_[\d]{4}-2d\.dat'
    f_name_re = re.compile(f_name_pattern)

    def name_match(pattern, name):
        """Asserts that the regexp matches the entire name"""
        match = pattern.match(name)
        if match:
            return match.end() == len(name)
        else:
            return False

    fingerprint_paths = [os.path.join(directory, f)
                         for f in os.listdir(directory)
                         if name_match(f_name_re, f)]

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
    """Load stability scores from CSV

    Parameters
    ----------
    score_path: Str
        Path to score .CSV file

    Returns
    -------
    Mapping[Str, Int]
        Maps base structure names to stability scores
    List[Str]
        List of entries with no stability data
    """
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


def get_stab_score(score_table, name):
    """Extract score from name->score mapping
    Fails gracefully if name is not in mapping
    """
    try:
        return score_table[name]
    except KeyError:
        print(name + " not in table")
        return None


def extract_topology(f_name):
    """Infers topology from filename and returns a code in [0,3]"""
    if f_name.startswith("HHH_"):
        topology = 0
    elif f_name.startswith("EHEE_"):
        topology = 1
    elif f_name.startswith("HEEH_"):
        topology = 2
    elif f_name.startswith("EEHEE_"):
        topology = 3
    else:
        raise ValueError("Invalid name {}".format(f_name))
    return topology


def load_fingerprint(f_name, n=100):
    """Loads a fingerprint from a dat file.
    Returns a `[n, n, 1]` array"""
    print_data = np.loadtxt(f_name, np.float32).reshape(n, n, 1)
    return np.flipud(np.transpose(print_data, [1, 0, 2]))


# reloads data from CSV and integration files
# saves to a numpy archive
def load_data_from_raws():
    """Load data from sources and save in archive"""
    import shutil
    import errno

    print("reloading raw data")
    archive_dir = os.path.abspath(config.archive_dir)

    # remove work directory (to clear any old sessions, etc)
    try:
        shutil.rmtree(archive_dir)
    except OSError as e:
        if e.errno == errno.ENOENT:
            pass
        else:
            raise

    # regenerate work dir
    os.makedirs(archive_dir)

    fingerprint_names = get_fingerprint_filenames(config.fingerprint_dir)
    stability_table, no_data = load_stab_scores(config.score_path)

    design_names = []
    fingerprints = []
    stab_scores = []
    topos = []

    # Gather data from files
    for f_name in fingerprint_names:
        if f_name not in no_data:
            design_name = get_design_name(f_name)
            score = get_stab_score(stability_table, design_name)
            fingerprint_arr = load_fingerprint(f_name)
            topology = extract_topology(design_name)

            if score is not None:
                design_names.append(design_name)
                fingerprints.append(fingerprint_arr)
                stab_scores.append(score)
                topos.append(topology)
        else:
            print("no data for " + f_name)

    # Concatenate data into arrays
    names_all = np.array(design_names, dtype='U15')
    prints_all = np.array(fingerprints, dtype=np.float32)
    scores_all = np.array(stab_scores, dtype=np.float32)
    topos_all = np.array(topos, dtype=np.int32)

    # save full files
    np.savez(os.path.join(archive_dir, config.full_archive_name),
             **{'names': names_all, 'fingerprints': prints_all, 'scores': scores_all, 'topologies': topos_all})

    print("raw data loaded and saved")
