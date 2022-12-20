# DataPreprocessing/utilities.py
# """
# # Graph Neural Network for Simulated X-Ray Transient Detection
# The present work aims to train a GNN to label a particular sort of X-Ray transient using simulated events 
# overlayed onto real data from XMM-Newton observations. We will experiment with Graph Convolutional Networks (GCNs).
# We will therefore  have to trandsform our point-cloud data into a "k nearest neighbors"-type graph. 
# Data stored in the `raw` folder at the current working directory is taken from icaro.iusspavia.it 
# `/mnt/data/PPS_ICARO_SIM2`. Observations store data for each photon detected, with no filter applied, 
# in FITS files ending in `EVLI0000.FTZ` for the original observations and `EVLF0000.FTZ` for the observation 
# and simulation combined. We will refer to the former data as "genuine" and to the latter as "faked" for brevity.
# """
"""
Utilities for reading event files
"""
import numpy as np

from astropy.table import Table
from astropy.table.operations import _join
from astropy import units as u


newunits = [u.def_unit("PIXELS", u.pixel),
            u.def_unit("CHAN", u.chan),
            u.def_unit("CHANNEL", u.chan),
            u.def_unit("0.05 arcsec", 0.05*u.arcsec)
           ]

def setdiff_idx(table1, table2, keys=None):
    """
    Take a set difference of table rows.

    The row set difference will contain all row indices in ``table1`` that are not
    present in ``table2``. If the keys parameter is not defined, all columns in
    ``table1`` will be included in the output table.

    Parameters
    ----------
    table1 : `~astropy.table.Table`
        ``table1`` is on the left side of the set difference.
    table2 : `~astropy.table.Table`
        ``table2`` is on the right side of the set difference.
    keys : str or list of str
        Name(s) of column(s) used to match rows of left and right tables.
        Default is to use all columns in ``table1``.

    Returns
    -------
    idx : `list`
        List containing the set difference indices between tables. If the set
        difference is none, an empty list will be returned.

    Examples
    --------
    To get a set difference index list between two tables::

      >>> from astropy.table import setdiff, Table
      >>> t1 = Table({'a': [1, 4, 9], 'b': ['c', 'd', 'f']}, names=('a', 'b'))
      >>> t2 = Table({'a': [1, 5, 9], 'b': ['c', 'b', 'f']}, names=('a', 'b'))
      >>> print(t1)
       a   b
      --- ---
        1   c
        4   d
        9   f
      >>> print(t2)
       a   b
      --- ---
        1   c
        5   b
        9   f
      >>> print(setdiff_idx(t1, t2))
      [1]

      >>> print(setdiff(t2, t1))
      [1]
    """

    if keys is None:
        keys = table1.colnames

    # Check that all keys are in table1 and table2
    for tbl, tbl_str in ((table1, 'table1'), (table2, 'table2')):
        diff_keys = np.setdiff1d(keys, tbl.colnames)
        if len(diff_keys) != 0:
            raise ValueError("The {} columns are missing from {}, cannot take "
                             "a set difference.".format(diff_keys, tbl_str))

    # Make a light internal copy of both tables
    t1 = table1.copy(copy_data=False)
    t1.meta = {}
    t1.keep_columns(keys)
    t1['__index1__'] = np.arange(len(table1))  # Keep track of rows indices

    # Make a light internal copy to avoid touching table2
    t2 = table2.copy(copy_data=False)
    t2.meta = {}
    t2.keep_columns(keys)
    # Dummy column to recover rows after join
    t2['__index2__'] = np.zeros(len(t2), dtype=np.uint8)  # dummy column

    t12 = _join(t1, t2, join_type='left', keys=keys,
                metadata_conflicts='silent')

    # If t12 index2 is masked then that means some rows were in table1 but not table2.
    if hasattr(t12['__index2__'], 'mask'):
        # Define bool mask of table1 rows not in table2
        diff = t12['__index2__'].mask
        # Get the row indices of table1 for those rows
        idx = t12['__index1__'][diff]
    else:
        idx = []

    return idx

#["TIME", "X", "Y", "PI", "FLAG"]
def read_events(genuine, simulated, keys):
    """
    Reads events from a genuine file and a simulated file, uses 'setdiff_idx' 
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
    
    D_dat_idx = setdiff_idx(F_dat, I_dat, keys=keys)
    
    dat = F_dat
    dat["ISFAKE"] = np.zeros(len(dat), dtype=bool)
    dat["ISFAKE"][D_dat_idx] = True
    keys += "ISFAKE"
    return dat[keys, "ISFAKE"]