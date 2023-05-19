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

def plot_fits_data(filename):
    # Read the FITS file and extract the data
    with fits.open(filename) as hdul:
        data = hdul[1].data
    
    # Extract the individual columns
    x = data['X']
    y = data['Y']
    time = data['TIME']
    pi = data['PI']
    is_event = data['ISEVENT']
    
    # Separate the data points based on the label (background or event)
    background_points = data[is_event == 0]
    event_points = data[is_event == 1]
    
    # Set the size of the points based on PI
    background_sizes = np.clip(background_points['PI'], 1, 100)
    event_sizes = np.clip(event_points['PI'], 1, 100)
    
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
    
    # Add a legend
    ax.legend()
    
    # Show the plot
    plt.show()

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Plot FITS data in 3D')
parser.add_argument('filename', type=str, help='Path to the FITS file')
args = parser.parse_args()

# Call the plot_fits_data function with the provided filename
plot_fits_data(args.filename)