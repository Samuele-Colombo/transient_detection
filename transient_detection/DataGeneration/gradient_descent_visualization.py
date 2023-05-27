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
Script: 3D Graph Plotter

This script reads JSON data from a file and plots a 3D graph using the data. The JSON data should be formatted
in a specific way with fields such as 'Epoch', 'Validation', 'loss', 'true_positives', 'true_negatives',
'true_positives_analog', and 'true_negatives_analog'.

The script uses the 'argparse' module to handle command-line arguments. The JSON file name and the output file name
should be provided as command-line arguments when running the script.

The main function 'plot_3d_graph' loads the JSON data using pandas, processes it, and plots the 3D graph. The x-axis
represents 'true_positives_analog', the y-axis represents 'true_negatives_analog', and the z-axis represents 'loss'.
The graph contains two lines of different colors, one for the 'validation=False' data, and the other for the
'validation=True' data. The order of the data is determined by the 'Epoch' value.

Usage:
    python script_name.py json_file output_file

"""

import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def plot_3d_graph(json_file, output_file):
    """
    Plot a 3D graph from JSON data.

    Parameters
    ----------
    json_file : str
        The name of the JSON file containing the data.
    output_file : str
        The name of the output HTML file.

    Returns
    -------
    None
        This function does not return anything, it saves the plot as an interactive HTML file.

    """
    try:
        # Read JSON data into a DataFrame using pandas
        df = pd.read_json(json_file, lines=True)

        # Sort the data based on the 'Epoch' value
        df.sort_values(by='Epoch', inplace=True)

        # Separate data for validation=True and validation=False
        validation_true = df[df['Validation'] == 'True']
        validation_false = df[df['Validation'] == 'False']

        # Extract x, y, and z values for each dataset
        x_true_analog = validation_true['true_positives_analog'].str.split(' ', expand=True)[0].astype(float)
        y_true_analog = validation_true['true_negatives_analog'].str.split(' ', expand=True)[0].astype(float)
        z_true = validation_true['loss'].str.split(' ', expand=True)[0].astype(float)

        x_false_analog = validation_false['true_positives_analog'].str.split(' ', expand=True)[0].astype(float)
        y_false_analog = validation_false['true_negatives_analog'].str.split(' ', expand=True)[0].astype(float)
        z_false = validation_false['loss'].str.split(' ', expand=True)[0].astype(float)

        x_true = validation_true['true_positives'].str.split(' ', expand=True)[0].astype(float)
        y_true = validation_true['true_negatives'].str.split(' ', expand=True)[0].astype(float)

        x_false = validation_false['true_positives'].str.split(' ', expand=True)[0].astype(float)
        y_false = validation_false['true_negatives'].str.split(' ', expand=True)[0].astype(float)
    except Exception as e:
        if 'x_true_analog' not in locals():
            print("Failed to assign x_true_analog: ", validation_true['true_positives_analog'].str.split(' ', expand=True))
        if 'y_true_analog' not in locals():
            print("Failed to assign y_true_analog: ", validation_true['true_negatives_analog'].str.split(' ', expand=True))
        if 'z_true' not in locals():
            print("Failed to assign z_true: ", validation_true['loss'].str.split(' ', expand=True))
        if 'x_false_analog' not in locals():
            print("Failed to assign x_false_analog: ", validation_false['true_positives_analog'].str.split(' ', expand=True))
        if 'y_false_analog' not in locals():
            print("Failed to assign y_false_analog: ", validation_false['true_negatives_analog'].str.split(' ', expand=True))
        if 'z_false' not in locals():
            print("Failed to assign z_false: ", validation_false['loss'].str.split(' ', expand=True))
        if 'x_true' not in locals():
            print("Failed to assign x_true: ", validation_true['true_positives'].str.split(' ', expand=True))
        if 'y_true' not in locals():
            print("Failed to assign y_true: ", validation_true['true_negatives'].str.split(' ', expand=True))
        if 'x_false' not in locals():
            print("Failed to assign x_false: ", validation_false['true_positives'].str.split(' ', expand=True))
        if 'y_false' not in locals():
            print("Failed to assign y_false: ", validation_false['true_negatives'].str.split(' ', expand=True))
        print("Exception:", e)
        return

    lossfunc = lambda X, Y: 1 - X * Y

    # Create a 3D scatter plot for validation=True
    scatter_true_analog = go.Scatter3d(
        x=x_true_analog,
        y=y_true_analog,
        z=z_true,
        mode='lines',
        name='Validation (Analog)',
        line=dict(color='blue', width=3)
    )

    scatter_true = go.Scatter3d(
        x=x_true,
        y=y_true,
        z=lossfunc(x_true, y_true),
        mode='lines',
        name='Validation',
        line=dict(color='blue', width=3, dash='dash')
    )

    # Create a 3D scatter plot for validation=False
    scatter_false_analog = go.Scatter3d(
        x=x_false_analog,
        y=y_false_analog,
        z=z_false,
        mode='lines',
        name='Training (Analog)',
        line=dict(color='red', width=3)
    )

    scatter_false = go.Scatter3d(
        x=x_false,
        y=y_false,
        z=lossfunc(x_false, y_false),
        mode='lines',
        name='Training',
        line=dict(color='red', width=3, dash='dash')
    )

    # Create a meshgrid for the function
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = lossfunc(X, Y)

    # Create a surface plot for the function
    surface = go.Surface(x=X, y=Y, z=Z, showscale=False, opacity=0.5)

    # Create the plot layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='True Positives'),
            yaxis=dict(title='True Negatives'),
            zaxis=dict(title='Loss')
        )
    )
    
    layout.update(title='Loss evolution of GCNN training and validation')

    # Create the figure
    fig = go.Figure(data=[scatter_true_analog, scatter_true, scatter_false_analog, scatter_false, surface], layout=layout)

    # Save the plot as an interactive HTML file
    fig.write_html(output_file)

    print(f"Plot saved as {output_file}")

if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Plot 3D graph from JSON data')

    # Add an argument for the JSON file name
    parser.add_argument('json_file', type=str, help='JSON file name')

    # Add an argument for the output file name
    parser.add_argument('output_file', type=str, help='Output file name')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the plot_3d_graph function with the provided JSON file name and output file name
    plot_3d_graph(args.json_file, args.output_file)
