import sys
import os
import os.path as osp
import subprocess
import argparse

parser = argparse.ArgumentParser(
    prog = 'install_requirements.py',
    description = "Using PIP, install all packages from 'requirements.txt'"
)
parser.add_argument("-o", "--options", action="extend", 
    nargs="*",
    help="Arguments are the `pip install` options without the initial double "+
         "dash (e.g. if I want `--dry-run` I add `dry-run` as argument. " +
         "Options with arguments must be enclosed between quotation marks " +
         "(e.g. `--root <dir>` is `'root <dir>'`). You should not change the " +
         " `--requirements` options. Single letter options cannot be bunched."
)

def to_options(opts):
    """Prepends '-' if the first element of a space separated string is a single letter; if a word, prepends '--'.

    Parameters
    ----------
    opts : list of str
        list of non-empty strings with at least one space-separated element.

    Returns
    -------
    list of str
        list of the given strings, `split`ted and `strip`ped of whitespaces and with the correct number of dashes prepended.
    """
    opts_out = []
    for opt in opts:
        assert len(opt) > 0, "Error: `to_option` encountered a zero-length string."
        head, *_ = opt_l = opt.strip().split()
        dashes = '-'
        if len(head) > 1:
            dashes += '-'
        opt_l[0] = dashes + head
        opts_out += opt_l
    return opts_out

parsed = parser.parse_args(sys.argv[1:])

if parsed.options is not None:
    options = to_options(parsed.options)
else:
    options = []

requirementsfile = osp.join(os.getcwd(), "requirements.txt")

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', *options,
'-r', requirementsfile])

import torch

packages=["pyg-lib", "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv"]

pygwheel = "https://data.pyg.org/whl/torch-{}.html".format(torch.__version__)

subprocess.check_call([sys.executable, '-m', 'pip', 'install', *options,
*packages, "-f", pygwheel])

subprocess.check_call([sys.executable, '-m', 'pip', 'install', *options,
"torch-geometric"])