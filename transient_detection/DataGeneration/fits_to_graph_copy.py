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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.io import fits

def generate_interactive_plot(filename, x_axis='X', y_axis='Y', z_axis=None):
    data = fits.getdata(filename)
    is_event = data['ISEVENT']

    if z_axis:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(data[x_axis], data[y_axis], data[z_axis], c=is_event, cmap='viridis')
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_zlabel(z_axis)
        plt.colorbar(scatter, label='ISEVENT')
    else:
        scatter = plt.scatter(data[x_axis], data[y_axis], c=is_event, cmap='viridis')
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.colorbar(scatter, label='ISEVENT')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive Plot Script')
    parser.add_argument('filename', help='Name of the FITS file to load.')
    parser.add_argument('--x_axis', default='X', help='Column to use as the x-axis. Default is "X".')
    parser.add_argument('--y_axis', default='Y', help='Column to use as the y-axis. Default is "Y".')
    parser.add_argument('--z_axis', help='Column to use as the z-axis. If not specified, a 2D plot will be generated.')
    args = parser.parse_args()

    generate_interactive_plot(args.filename, args.x_axis, args.y_axis, args.z_axis)
