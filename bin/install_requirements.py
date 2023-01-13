#!python
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

import sys
import os
import os.path as osp
import tempfile
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        prog = 'install_requirements.py',
        description = "Using PIP, install all packages from 'requirements.txt'",
        prefix_chars="@"
    )
    parser.add_argument('@beluga', action='store_true', help="Specify if running on beluga so that specific requirements are met. Options are ignored.")
    parser.add_argument("options", action="store", 
        nargs="*", default=[],
        help="Add here any `pip install` options (e.g. if you want to make a dry run you may "+
            "add `--dry-run`as argument. You must not change the `-r | --requirements` option."
    )
    parsed = parser.parse_args(sys.argv[1:])

    options = parsed.options

    if parsed.beluga:
        subprocess.check_call(["pip", "install", "--no-index", "-r", "requirements.beluga.txt"])
        return

    if '-r' in options or "--requirements" in options:
        raise Exception("Error: you must not modify the `-r | --requirements` option.")

    requirementsfile = osp.join(os.getcwd(), "requirements.txt")

    # List of packages to exclude
    excluded_packages = ["torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric"]

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)

    # Open the requirements file
    with open(requirementsfile, 'r') as f:
        # Iterate through each line in the file
        for line in f:
            # Check if the line contains one of the excluded packages
            if any(package in line for package in excluded_packages):
                # Skip this line
                continue
            # Write the line to the temporary file
            temp_file.write(line)

    # Close the file
    temp_file.close()

    # Install the packages listed in the temporary file
    subprocess.run(['pip', 'install', *options,  '-r', temp_file.name])

    # Delete the temporary file
    os.unlink(temp_file.name)

    ## BUG: problems when deployed in SLURM. requests.get raises ConnectionError
    import torch
    import requests

    this_version = torch.__version__
    pytorch_major_version, pytorch_minor_version, pytorch_patch_version = \
        this_version.split("+")[0].split(".")
    try:
        pytorch_cuda_version = this_version.split("+")[1]
    except IndexError:
        pytorch_cuda_version = "cpu"
        this_version = "+".join([this_version, pytorch_cuda_version])

    while int(pytorch_patch_version) >= 0:
        pygwheel_url = "https://data.pyg.org/whl/torch-{}.html".format(this_version)
        r = requests.get(pygwheel_url)
        if r.status_code == 200:
            # The webpage exists, so we can exit the loop
            print(f"- Installing `pytorch_geometric` packages from wheel for version {this_version}.")
            break
        else:
            # The webpage does not exist, so we reduce the patch version of PyTorch and try again
            pytorch_patch_version = str(int(pytorch_patch_version) - 1)
            print(f"- Failed to recover a wheel for `pytorch_geometric` for version {this_version}, trying for previous patch versions...", file=sys.stderr)
            this_version = "{major}.{minor}.{patch}+{cuda}".format(
                major = pytorch_major_version,
                minor = pytorch_minor_version,
                patch = pytorch_patch_version,
                cuda  = pytorch_cuda_version
            )
    else:
        raise RuntimeError(f"Error: could not find a suitable wheel for torch version {torch.__version__}")


    pygwheel = "https://data.pyg.org/whl/torch-{}.html".format(this_version)

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', *options,
    *excluded_packages, "-f", pygwheel])

if __name__ == "__main__":
    main()