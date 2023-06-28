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
import argparse
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from transient_detection.DataPreprocessing.utilities import read_events

def plot_fits_data(filename, outfile, save=True):
    # Read the FITS file and extract the data
    print("Opening file...")
    with fits.open(filename) as hdul:
        data = hdul[1].data
    
    if 'ISEVENT' not in data.columns.names:
        if 'EVLI' in filename:
            companion = filename.replace('EVLI', 'EVLF')
            data =read_events(filename, companion, ['X', 'Y', 'TIME', 'PI'])
        elif 'EVLF' in filename:
            companion = filename.replace('EVLI', 'EVLF')
            data =read_events(companion, filename, ['X', 'Y', 'TIME', 'PI'])
        else:
            raise Exception("filename does not contain the 'ISEVENT' colname and has not the 'EVLI' or 'EVLF' indicator in the file name. Check file integrity")
        data.rename_column('ISSIMULATED', 'ISEVENT')

    data['PI'] = np.log2(((data['PI'] - data['PI'].min())/(data['PI'].max() - data['PI'].min())) + 2) * 10

    # Extract the individual columns
    is_event = data['ISEVENT']
    
    # Separate the data points based on the label (background or event)
    background_points = data[is_event == 0]
    event_points = data[is_event == 1]
    
    # Set the size of the points based on PI
    background_sizes = background_points['PI']
    event_sizes = event_points['PI']

    print("Plotting...")
    # Create the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the background points
    ax.scatter(background_points['X'], background_points['TIME'], background_points['Y'],
               c='b', label='Background', alpha=0.6, s=background_sizes)
    
    # Plot the event points
    ax.scatter(event_points['X'], event_points['TIME'], event_points['Y'],
               c='r', label='Event', alpha=0.6, s=event_sizes)
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Time')
    ax.set_zlabel('Y')

    import os.path as osp
    
    ax.set_title(f"Transient from '{osp.basename(filename)}'\n {len(background_points)} bkg events, {len(event_points)} transient events.")

    # Add a legend
    ax.legend()
    
    # Show the plot
    if save:
        plt.savefig(outfile)
    else:
        plt.show()

if __name__ == "__main__":
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='Plot FITS data in 3D')
    parser.add_argument('filename', type=str, help='Path to the FITS file')
    parser.add_argument('outfile', type=str, help='Path to the output image file')
    args = parser.parse_args()

    # Call the plot_fits_data function with the provided filename
    plot_fits_data(args.filename, args.outfile)
