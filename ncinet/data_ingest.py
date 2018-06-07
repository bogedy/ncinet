"""
Ingest data and store it in numpy arrays.
"""

import os
import numpy as np
import pandas as pd

from typing import Sequence, Tuple, Mapping, MutableMapping
from ncinet.config_meta import DataIngestConfig, PredictIngestConfig


# ------------------------------------
# Ingest framework for original Rocklin data
# ------------------------------------

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


def load_rocklin(config):
    # type: (DataIngestConfig) -> Mapping[str, np.ndarray]
    """Load data present in the data format of the original Rocklin data.
    NCI fingerprints are read from individual files, stabilities are read from
    a CSV, topologies are derived from the file name
    """
    fingerprint_names = get_fingerprint_filenames(config.nci_dir)
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

    return {'names': names_all, 'fingerprints': prints_all, 'scores': scores_all, 'topologies': topos_all}


# ------------------------------------
# Ingest from preprocessed DataFrames
# ------------------------------------

def topo_from_dssp(dssp):
    # type: (str) -> str
    """Converts a DSSP string to a topology label."""
    secondary_list = []
    last = None
    for x in dssp.upper():
        assert x in "LEH"
        if x != last:
            if x != 'L':
                secondary_list.append(x)
            last = x

    return "".join(secondary_list)


def index_topologies(*topo_arrays):
    # type: (*np.ndarray) -> Tuple[str]
    """Find the unique topologies in a set of arrays."""
    return tuple(np.unique(np.concatenate(topo_arrays)))


def apply_topo_index(topo_arr, topo_index):
    # type: (np.ndarray, Sequence[str]) -> np.ndarray
    """Replace string topology labels with integer indices."""
    topo_map = {t: i for i, t in enumerate(topo_index)}
    return np.array([topo_map[x] for x in topo_arr])


def read_topo_labels(path):
    # type: (str) -> Tuple[str]
    """Loads a topo labeling from file."""
    import yaml
    with open(path) as label_file:
        topo_index = yaml.load(label_file)['topology index']
        return tuple(topo_index.keys())


def write_topo_labels(path, topo_index):
    # type: (str, Tuple[str]) -> None
    """Writes a topology labeling to file."""
    import yaml
    topo_map = {t: i for i, t in enumerate(topo_index)}
    with open(path, 'w') as label_file:
        yaml.dump({'topology index': topo_map}, label_file)


def join_fingerprint_df(nci_df_path, base_df):
    # type: (str, pd.DataFrame) -> pd.DataFrame
    """Load a DataFrame of NCI features and joins it to a template df.

    Parameters
    ----------
    nci_df_path: path
        Path to DataFrame with NCI data. Should be in a TSV format. Must have
        `name` and `library` columns, and NCI columns of `nci1` ... `nci10000`
    base_df: DataFrame
        DataFrame to join with NCI data, must have `name` column.
    """
    print("Loading NCI data from '{}'".format(os.path.basename(nci_df_path)))

    nci_df = pd.read_table(nci_df_path)
    merged_df = pd.merge(base_df, nci_df, on='name')

    print("Using {}/{} records from '{}' library.".format(len(merged_df), len(nci_df), nci_df.loc[0, 'library']))
    return merged_df.drop('library', axis=1)


def load_data_from_tables(df_path, nci_dir, expect_scores=True):
    # type: (str, str, bool) -> MutableMapping[str, np.ndarray]
    """Merges NCI data from DataFrames with topology and stability data from another.

    Parameters
    ----------
    df_path: path
        Path to the score and topology DataFrame. Should be in CSV format with
        `name`, `dssp`, and possibly `stabilityscore` and `stable?` columns.
    nci_dir: path
        Directory to search for NCI DataFrames.
    expect_scores: bool
        Whether to expect stability scores.
    """

    # Load the DataFrame specifying scores and topologies
    columns = ['name', 'dssp', 'stabilityscore', 'stable?'] if expect_scores else ['name', 'dssp']
    main_df = pd.read_table(df_path, sep=',', usecols=columns)
    main_df.loc[:, 'dssp'] = main_df.loc[:, 'dssp'].map(topo_from_dssp)
    main_df = main_df.rename(columns={'dssp': 'topologies', 'stabilityscore': 'scores'})

    def canonicalize_name(name):
        """Resolves differences in naming between datasets."""
        if name.endswith('.seq'):
            i = name.rfind('.pdb')
            return name[0:i] + '.pdb'
        if not name.endswith('.pdb'):
            return name + '.pdb'
        else:
            return name

    main_df.loc[:, 'name'] = main_df.loc[:, 'name'].map(canonicalize_name)

    # Make individual dfs based on each NCI archive
    per_library_dfs = []
    for nci_file in os.listdir(nci_dir):
        if not nci_file.endswith('.tsv.gz'):
            continue

        per_library_dfs.append(join_fingerprint_df(os.path.join(nci_dir, nci_file), main_df))

    # Concatenate individual dfs
    joined_df = pd.concat(per_library_dfs, axis=0, ignore_index=True)
    joined_df = joined_df.rename(columns={'name': 'names'})

    print("Using {}/{} records from '{}'".format(len(joined_df), len(main_df), os.path.basename(df_path)))

    # Extract and reshape nci data, assumes 100 x 100 fingerprints
    nci_labels = ['nci{}'.format(i) for i in range(1, 10001)]
    nci_data = joined_df.loc[:, nci_labels].values.reshape((-1, 100, 100, 1))

    # Extract other data from df
    output_cols = ['names', 'scores', 'topologies', 'stable?'] if expect_scores else ['names', 'topologies']
    output_dict = {c_name: joined_df.loc[:, c_name].values for c_name in output_cols}
    output_dict['fingerprints'] = nci_data

    return output_dict


# ------------------------------------
# Main dispatch method for ingest pipeline
# ------------------------------------

def load_data_from_raws(config):
    # type: (DataIngestConfig) -> None
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

    # Decide whether to use old or new ingest version
    if config.ingest_version == 'rocklin_v1':
        # Load data from fingerprint files and save to archive
        np.savez(os.path.join(archive_dir, config.full_archive_name),
                 **load_rocklin(config))

    elif config.ingest_version == 'sd2_dataframes_v2':
        # Check whether we are using pre-split data
        if type(config.score_path) is str:
            full_data = load_data_from_tables(config.score_path, config.nci_dir)

            # Convert topology tags to indices
            topo_index = index_topologies(full_data['topologies'])
            write_topo_labels(os.path.join(config.archive_dir, config.topo_index_name), topo_index)
            full_data['topologies'] = apply_topo_index(full_data['topologies'], topo_index)

            # Save data to file
            np.savez(os.path.join(archive_dir, config.full_archive_name), **full_data)
        elif type(config.score_path) is tuple:
            assert len(config.score_path) == 2

            # Load train and test data
            train_path, test_path = config.score_path
            train_data = load_data_from_tables(train_path, config.nci_dir)
            test_data = load_data_from_tables(test_path, config.nci_dir)

            # Convert topologies to indices
            topo_index = index_topologies(train_data['topologies'], test_data['topologies'])
            write_topo_labels(os.path.join(config.archive_dir, config.topo_index_name), topo_index)
            train_data['topologies'] = apply_topo_index(train_data['topologies'], topo_index)
            test_data['topologies'] = apply_topo_index(test_data['topologies'], topo_index)

            # Save data to files
            name_fstring = "raw_{prefix}_{{batch}}.npz".format(prefix=config.archive_prefix)
            np.savez(os.path.join(config.archive_dir, name_fstring.format(batch=config.tt_tags[0])), **train_data)
            np.savez(os.path.join(config.archive_dir, name_fstring.format(batch=config.tt_tags[1])), **test_data)
        else:
            raise ValueError("Unexpected value for 'score_path'")
    else:
        raise ValueError

    print("raw data loaded and saved")


def load_prediction_data(config):
    # type: (PredictIngestConfig) -> None
    """Load prediction data from tables and write to archive."""
    # Load data from raws
    predict_data = load_data_from_tables(config.dataframe_path, config.nci_dir, expect_scores=False)
    topo_index = read_topo_labels(os.path.join(config.archive_dir, config.topo_index_name))

    # Apply topo transform
    predict_data['topologies'] = apply_topo_index(predict_data['topologies'], topo_index)

    # Save archive
    np.savez(os.path.join(config.archive_dir, config.archive_name), **predict_data)
