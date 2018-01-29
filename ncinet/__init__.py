"""
foo
"""

__version__ = "0.2.0"

import os

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


class training_config:
    use_learning_rate_decay = True
    num_epochs_per_decay = 350.0
    learning_rate_decay_factor = 0.05
    initial_learning_rate = 0.005
    batch_size = 32
    # TODO: this number is not quite right...
    num_examples_per_epoch_train = 12000
    log_frequency = 20
    max_steps = 100000
    checkpoint_secs = 100
    summary_steps = 50
    train_dir = ""
