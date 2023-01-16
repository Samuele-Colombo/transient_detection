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

import astropy.table as astropy_table

from astropy import units as u
from astropy.io import fits
from astropy.table import Table


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

    # Add a new column indicating whether the event is simulated
    dat['ISSIMULATED'] = astropy_table.Column([False] * len(I_dat) + [True] * len(F_dat), dtype=bool)

    # Select only the columns specified in the `keys` parameter and the "ISSIMULATED" column
    keys = list(keys) + ['ISSIMULATED']
    return dat[keys]
