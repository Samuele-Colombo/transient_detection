# Copyright (c) 2023-present Samuele Colombo
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
fits_to_csv.py
==============

This script converts a FITS file to a comma-separated values (CSV) file.

Usage:
------
python fits_to_csv.py <fits_file> <csv_file>

Arguments:
----------
<fits_file> : str
    Path to the input FITS file.
<csv_file> : str
    Path to the output CSV file.

Example:
--------
python fits_to_csv.py data.fits data.csv
    Convert data.fits to data.csv.

"""

import argparse
import astropy.io.fits as fits
import pandas as pd

def fits_to_csv(fits_file, csv_file):
    """
    Convert a FITS file to a comma-separated values (CSV) file.

    Parameters
    ----------
    fits_file : str
        Path to the input FITS file.
    csv_file : str
        Path to the output CSV file.

    Notes
    -----
    This function reads the input FITS file using the astropy.io.fits module and converts the data to a
    pandas DataFrame. It then saves the DataFrame as a CSV file using the pandas DataFrame's to_csv method.

    Example
    -------
    fits_to_csv('data.fits', 'data.csv')
        Convert data.fits to data.csv.
    """

    if fits_file == csv_file:
        raise ValueError("Input FITS file and output CSV file cannot have the same value.")

    # Read the FITS file
    with fits.open(fits_file) as hdul:
        data = hdul[1].data

    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(data)

    # Save the DataFrame as a CSV file
    df.to_csv(csv_file, index=False)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert FITS file to CSV file")
    parser.add_argument("fits_file", type=str, help="Path to the input FITS file")
    parser.add_argument("csv_file", type=str, help="Path to the output CSV file")
    args = parser.parse_args()

    # Convert the FITS file to CSV
    fits_to_csv(args.fits_file, args.csv_file)
