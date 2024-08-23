# -*- coding: utf-8 -*-
"""
Functions for loading test data sets
"""

import os.path as op

import sklearn.utils
from pkg_resources import resource_filename
import numpy as np
from sklearn.utils import Bunch




# TODO: ignore for now, add at the end for an example

_res_path = resource_filename('packages', 'snf/toy_data/{resource}')


def _load_data(dset: str, dfiles: list[str]) -> sklearn.utils.Bunch:
    """
    Loads `dfiles` for `dset` and return Bunch with data and labels

    Parameters
    ----------
    dset : {'sim', 'digits'}
        Dataset to load
    dfiles : list of str
        Data files in `dset`

    Returns
    -------
    data : :obj:`sklearn.utils.Bunch`
        With keys `data` and `labels`
    """

    dpath = _res_path.format(resource=dset)
    if not op.isdir(dpath):  # should never happen
        raise ValueError('{} is not a valid dataset. If you are receiving '
                         'this error after using snf.datasets.load_simdata() '
                         'or snf.datasets.load_digits() it is possible that '
                         'snfpy was improperly installed. Please check your '
                         'installation and try again.'.format(dset))

    # space versus comma-delimited files (ugh)
    try:
        data = [np.loadtxt(op.join(dpath, fn)) for fn in dfiles]
    except ValueError:
        data = [np.loadtxt(op.join(dpath, fn), delimiter=',') for fn in dfiles]

    return Bunch(data=data,
                 labels=np.loadtxt(op.join(dpath, 'label.csv')))


def load_simdata():
    """
    Loads "similarity" data with two datatypes

    Returns
    -------
    sim : :obj:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['data', 'labels']
    """

    dfiles = [
        'data1.csv', 'data2.csv'
    ]

    return _load_data('sim', dfiles)


def load_digits():
    """
    Loads "digits" dataset with four datatypes

    Returns
    -------
    digits : :obj:`sklearn.utils.Bunch`
        Dictionary-like object with keys ['data', 'labels']
    """

    dfiles = [
        'fourier.csv', 'pixel.csv', 'profile.csv', 'zer.csv'
    ]

    return _load_data('digits', dfiles)