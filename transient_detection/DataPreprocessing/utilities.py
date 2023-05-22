# DataPreprocessing/utilities.py
# Copyright (c) 2022-present Samuele Colombo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and 
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR 
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
This module contains utility functions for reading and processing event data from FITS files.

Functions
---------
read_events(genuine, simulated, keys)
    Reads events from a genuine file and a simulated file, and removes the duplicate events.
    The function also adds a new column indicating whether the event is simulated.

Examples
--------
# Import the utility functions
from transient_detection.DataPreprocessing import utilities

# Read events from a genuine file and a simulated file
events = utilities.read_events('genuine.fits', 'simulated.fits', ['X', 'Y', 'Z'])

# Print the table of events
print(events)

# Access the 'ISSIMULATED' column of the table
issimulated = events['ISSIMULATED']
print(issimulated)
"""

import socket
import errno

import astropy.table as astropy_table

from astropy import units as u
from astropy.io import fits
from astropy.table import Table

import torch

newunits = [u.def_unit("PIXELS", u.pixel),
            u.def_unit("CHAN", u.chan),
            u.def_unit("CHANNEL", u.chan),
            u.def_unit("0.05 arcsec", 0.05*u.arcsec)
           ]


def read_events(genuine, simulated, keys):
    """
    Reads events from a genuine file and a simulated file, and removes the duplicate events.
    The function also adds a new column indicating whether the event is simulated.

    Parameters
    ----------
    genuine : str, file-like object, list, pathlib.Path object
        File containing only observed events.
    simulated : str, file-like object, list, pathlib.Path object
        File containing observed events plus simulated ones.
    keys : str or list of str
        Column labels for the attributes of interest

    Returns
    -------
    astropy.Table
        A table containing the attributes selected through the `keys` parameter
        and the "ISSIMULATED" value for each event, indicating whether the event is simulated or not.
    """
    # Read the genuine and simulated events
    with u.add_enabled_units(newunits), \
         fits.open(genuine, memmap=True) as gen_file, \
         fits.open(simulated, memmap=True) as sim_file:
        I_dat = Table(gen_file[1].data)
        F_dat = Table(sim_file[1].data)

    # Join the genuine and simulated events and remove the duplicate events
    dat = astropy_table.join(I_dat, F_dat, keys=keys, join_type='outer')
    dat = astropy_table.unique(dat, keys=keys, keep='first')

    num_simulated = len(dat) - len(I_dat)

    # Add a new column indicating whether the event is simulated
    # Simulated events are all last since `F_dat` was appended and any
    # non-simulated event in it would have been discarded by the `keep='first'`
    # argumento of `astropy_table.unique`
    dat['ISSIMULATED'] = astropy_table.Column([False] * len(I_dat) + [True] * num_simulated, dtype=bool)

    # Select only the columns specified in the `keys` parameter and the "ISSIMULATED" column
    keys = list(keys) + ['ISSIMULATED']
    return dat[keys]

from glob import glob
import os.path as osp

def get_paired_filenames(raw_dir, genuine_pattern, simulated_pattern):
    g_names = glob(osp.join(raw_dir, genuine_pattern))
    g_ending = genuine_pattern.split('*')[-1]
    s_ending = simulated_pattern.split('*')[-1]
    for g_name in g_names:
        s_name = g_name.replace(g_ending, s_ending)
        if osp.isfile(s_name):
            yield g_name, s_name

import numpy as np

def in2d(a, b):
    dtype=a.dtype
    return np.in1d(a.view(dtype='{0},{0}'.format(dtype)).reshape(a.shape[0]),b.view(dtype='{0},{0}'.format(dtype)).reshape(b.shape[0]))

def get_uncompliant(compliance_file):
    with open(compliance_file, 'r') as f:
        for line in f:
            yield tuple(line.split())

class StandardScaler():
    def __init__(self) -> None:
        pass

    def fit(self, tensor: torch.tensor) -> None:
        self.m = tensor.mean(0, keepdim=True)
        self.s = tensor.std(0, unbiased=False, keepdim=True)
    
    def transform(self, tensor: torch.tensor) -> torch.tensor:
        tensor -= self.m
        tensor /= self.s
        return tensor

def is_socket_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except socket.error as e:
            if e.errno == errno.EADDRINUSE:
                return False
            else:
                raise
        else:
            return True
