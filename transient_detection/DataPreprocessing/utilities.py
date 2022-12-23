# DataPreprocessing/utilities.py
import numpy as np

from astropy.table import Table
from astropy.table.operations import _join
from astropy import units as u


newunits = [u.def_unit("PIXELS", u.pixel),
            u.def_unit("CHAN", u.chan),
            u.def_unit("CHANNEL", u.chan),
            u.def_unit("0.05 arcsec", 0.05*u.arcsec)
           ]

def find_missing_rows(table1, table2, keys=None):
    """
    Find the indices of rows in `table1` that are not present in `table2` by comparing the values in the rows indexed by the given keys. If `keys` is `None`, the whole rows are compared.

    This function uses a hash-based search to improve the runtime complexity. The runtime complexity is O(n + m), where n is the number of rows in `table1` and m is the number of rows in `table2`.

    Parameters
    ----------
    table1 : astropy.table.Table
        The first table.
    table2 : astropy.table.Table
        The second table.
    keys : list of str, optional
        A list of keys to use for comparison. If `None`, the whole rows are compared. Default is `None`.

    Returns
    -------
    list of int
        A list of indices of rows in `table1` that are not present in `table2`.
    """
    if keys is None:
        # Create a dictionary with the rows in table2 as keys and their indices as values
        table2_dict = {row: i for i, row in enumerate(table2)}

        # Find the indices of all the rows in table1 that are not present in table2
        missing_rows = [i for i, row in enumerate(table1) if row not in table2_dict]
    else:
        # Create a dictionary with the rows in table2 indexed by the given keys as keys and their indices as values
        table2_dict = {tuple(row[key] for key in keys): i for i, row in enumerate(table2)}

        # Find the indices of all the rows in table1 that are not present in table2
        missing_rows = [i for i, row in enumerate(table1) if tuple(row[key] for key in keys) not in table2_dict]

    return missing_rows




#["TIME", "X", "Y", "PI", "FLAG"]
def read_events(genuine, simulated, keys):
    """
    Reads events from a genuine file and a simulated file, uses 'find_missing_rows' 
    to label events present in the simulated file and not in the genuine one
    in a new column "ISFAKE".

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
        and the "ISFAKE" value for each event in the simulated file.
    """
    u.add_enabled_units(newunits)
    
    I_dat = Table.read(genuine, hdu=1)
    F_dat = Table.read(simulated, hdu=1)
    
    D_dat_idx = find_missing_rows(F_dat, I_dat, keys=keys)
    
    dat = F_dat
    dat["ISFAKE"] = np.zeros(len(dat), dtype=bool)
    dat["ISFAKE"][D_dat_idx] = True
    keys += "ISFAKE"
    return dat[keys, "ISFAKE"]