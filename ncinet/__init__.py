"""
foo
"""

__version__ = "0.2.0"


# Directory where summaries and checkpoints are written.
#WORK_DIR = '/work/05187/ams13/maverick/Working/TensorFlow/ncinet'
WORK_DIR = 'C:/Users/schan/Documents/TF_run'

FINGERPRINT_DIR = '/work/projects/SD2E-Community/prod/data/shared-q0-hackathon/team10/nci_rocklin'


class data_config:
    archive_dir = WORK_DIR
    fingerprint_dir = FINGERPRINT_DIR
    score_path = os.path.join(WORK_DIR, '../output.csv')
    full_archive_name = "data_full.npz"
    archive_prefix = "data"
    tt_tags = ("train", "eval")
    n_folds = 5
    xv_tags = ("xvTrain", "xvVal")
