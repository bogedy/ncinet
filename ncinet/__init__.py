"""
Module to build and train neural networks to predict protein properties.
"""

import os
import yaml

__version__ = "0.6.0"

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
BASE_CONF_FILE = os.path.join(CONFIG_DIR, "base_config.yml")

with open(BASE_CONF_FILE, 'r') as base_conf_file:
    BASE_CONFIG = yaml.load(base_conf_file)

# Directory where summaries and checkpoints are written.
WORK_DIR = BASE_CONFIG.pop('work_dir')
