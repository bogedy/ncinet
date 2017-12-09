"""
Call ncinet using this script. The script adds the ncinet root
directory onto the python path, and then passes control to
``ncinet.cli``. The majority of the UI is implemented in ``cli``.
"""

import os
import sys

# most of this is derived from bumps
# https://github.com/bumps/bumps


def add_path(path):
    """
    Add a directory to the python path environment, and
    to the PYTHONPATH environment variable for subprocesses.
    """
    path = os.path.abspath(path)
    if 'PYTHONPATH' in os.environ:
        PYTHONPATH = path + os.pathsep + os.environ['PYTHONPATH']
    else:
        PYTHONPATH = path
    os.environ['PYTHONPATH'] = PYTHONPATH
    sys.path.insert(0, path)


def prepare_env():
    root = os.path.abspath(os.path.dirname(__file__))
    add_path(root)


if __name__ == '__main__':
    prepare_env()
    import ncinet.cli
    ncinet.cli.cli()
