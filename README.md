# NCInet

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/aschankler/ncinet)
[![GitHub tag](https://img.shields.io/github/tag/aschankler/ncinet.svg)](https://github.com/aschankler/ncinet/releases)
[![license](https://img.shields.io/github/license/aschankler/ncinet.svg)](https://github.com/aschankler/ncinet/blob/master/LICENSE)

> Convolutional neural network to predict protein stability from non-covalent index plots.

NCInet is part of a project to predict the stability of _de novo_ protein designs. It approaches this problem by learning from representations of the noncovalent interactions in the proteins.

This project implements a convolutional neural network that takes as input two dimensional plots of the reduced density gradient (RDG) of the electron density of a protein. The use of the reduced density gradient to encode bonding information is discussed by [Johnson et al.](http://pubs.acs.org/doi/abs/10.1021/ja100936w) The experimental stability measurements used for training are published by [Rocklin et al.](http://www.sciencemag.org/lookup/doi/10.1126/science.aan0693) though training has also been performed on more extensive datasets.

NCInet was initially developed by Aaron Schankler as part of an undergraduate thesis at Haverford College. The project was advised by Joshua Schrier.


## Installation

After cloning the repository, edit the paths in `ncinet/config/base_config.yml`.

### Requirements
The exact version requirements are not known. The program is known to work with the versions listed, but earlier or later versions may also work.

* Python 3 (tested on `3.5.2`)
* numpy `1.11.0`
* pandas `0.18.1`
* TensorFlow `1.0.0`
* pyyaml `3.12`


## Usage

NCInet provides an API in `ncinet.names`, but is intended to be used primarily through the command line.

### CLI

The program is run with `python3 run.py <args>`. The arguments specify the type of model being run and what should be done on that model.

One of the following must be specified:

* `--train` to train the loaded model.
* `--eval` to load a model from a checkpoint and evaluate it on held-out data.
* `--xval` to cross validate the chosen hyperparameters using the training dataset.
* `--grid CONFIG` and `--rand CONFIG N_RUNS` to run hyperparameter optimization

Models can be specified with the following commands:
* `--ae` loads the default autoencoder configuration.
* `--sign`, `--stable`, and `--topo` load the default classifiers of sign, stability, and topology respectively.
* `--conf CONFIG_FILE` allows the default configuration file to be updated.
* `--basename NAME` affects the directory from which checkpoints are saved and loaded.
* `--out_file NAME` specifies a file to write hyperparameter optimization output

More complete information is available with `python3 run.py --help`

### Config files

NCInet uses config files, both from the `ncinet/config` directory and from locations supplied with `--conf` and `--rand` arguments. Config files are written in the [YAML](http://yaml.org/spec/1.1/#id857168) data serialization language and are used to generate config object (which are defined in `config_meta.py`). The configuration is initialized from the primary configuration found in `base_config.yml` and then updated from user specified secondary config files.

The config files must decode to a mapping with the keys `ingest_config`, `request_config`, `training_config`, and `eval_config`. Secondary config files must also contain a `model_config` key. Each of these keys maps to a (possibly empty) mapping whose keys in turn correspond to attributes of the associated config object. Documentation for these attributes is provided in `config_meta.py`. Unexpected keys are disallowed.

The primary config file also contains a `work_dir` key, which specifies the default path to search for and save checkpoint directories. Secondary config files must include a `model_type` key, which maps to a string selecting which model to run. Currently accepted models are `encoder`, `topo`, `sign`, and `stable`.

Files for hyperparameter optimization (supplied with `--rand` and `--grid`) have a sightly different format. The files decode to a mapping with keys `fixed_params` and `var_params`. The first of these keys maps to a secondary configuration as described above. The second maps to a dictionary whose keys are python tuples. The `var_params` mapping is used to create many different versions of the secondary configuration.


## Maintainers

* [Aaron Schankler](https://github.com/aschankler)

## License
[MIT](https://choosealicense.com/licenses/mit/) &copy; Aaron Schankler. For more information, [see `LICENSE`](https://github.com/aschankler/ncinet/blob/master/LICENSE).
