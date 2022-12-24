# DataPreprocessing/utilities.py
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


from astropy import units as u
import astropy.table as astropy_table


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
    with u.add_enabled_units(newunits):
        I_dat = astropy_table.Table.read(genuine, hdu=1)
        F_dat = astropy_table.Table.read(simulated, hdu=1)

    # Join the genuine and simulated events and remove the duplicate events
    dat = astropy_table.join(I_dat, F_dat, keys=keys, join_type='outer')
    dat = astropy_table.unique(dat, keys=keys, keep='first')

    # Add a new column indicating whether the event is simulated
    dat['ISSIMULATED'] = astropy_table.Column([False] * len(I_dat) + [True] * len(F_dat), dtype=bool)

    # Select only the columns specified in the `keys` parameter and the "ISSIMULATED" column
    keys = list(keys) + ['ISSIMULATED']
    return dat[keys]
