# Deeplearning/utilities.py
"""
Utilities for logging training data
"""

import os.path as osp
import sys
import yaml

def log(logfile, label, forcemode=None, **loggings):
    """
    Logs given values to a logfile in YAML format.

    Parameters
    ----------
    logfile : str
        relative or absolute path to a log file. If it does not exist it is created, 
        else the behavior is dictated by user input or forcemode.

    label : str
        a label which will be used as header for the log entry (e.g. "epoch 1").

    forcemode : str, optional
        if not `None`, forces write mode. Accepts 'w', 'a' and None as values. 
        Value 'w' overwrites target file. Value 'a' appends to target file. By default None
    
    **loggings : key=value pairs
        what is being logged.
    """
    if not forcemode is None:
        assert forcemode in ["w", "a"], f"Error: `forcemode` is '{forcemode}'. Must be either 'w' or 'a'"
    loggings = {label: loggings}
    yaml.dump(loggings, sys.stderr)
    mode = "w+"
    if osp.exists(logfile) and forcemode is None:
        usrinpt=""
        while not usrinpt in ["O","E","C"]:
            usrinpt = input(f"Do you want to overwrite [O] or extend [E] already existing log file {logfile}? (C to cancel) [O,E,C] ")
        if usrinpt == "C":
            return
        elif usrinpt == "E":
            mode = "a"
    elif not forcemode is None:
        mode = forcemode
    with open(logfile, mode) as lf:
        #print(*(f"{key}: {value}" for key, value in loggings.items()), sep="\n\t", file=lf)
        yaml.dump(loggings,lf)