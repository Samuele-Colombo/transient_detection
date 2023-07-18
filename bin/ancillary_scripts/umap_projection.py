import pandas as pd
import umap
import matplotlib.pyplot as plt
from astropy.io import fits
from transient_detection.DataPreprocessing.utilities import read_events
import argparse

def load_fits_to_dataframe(filename):
    # Load the FITS file into a Pandas DataFrame
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
    data = data.to_pandas()
    return data

def apply_umap_projection(data):
    # Select the desired columns from the DataFrame
    columns = ["TIME", "X", "Y", "PI", "ISEVENT"]
    selected_data = data[columns]

    # Separate data points and their corresponding "ISEVENT" labels
    X = selected_data.drop("ISEVENT", axis=1)
    y = selected_data["ISEVENT"]

    # Apply UMAP projection
    reducer = umap.UMAP()
    umap_projection = reducer.fit_transform(X)

    return umap_projection, y

def plot_umap(filename, plot_name, save=True):
    # Load FITS file into DataFrame
    fits_data = load_fits_to_dataframe(filename)

    # Apply UMAP projection and get "ISEVENT" labels
    umap_projection, is_event_labels = apply_umap_projection(fits_data)
    
    # Plot the UMAP projection with different colors for "ISEVENT" labels
    plt.scatter(umap_projection[is_event_labels == True, 0], umap_projection[is_event_labels == True, 1], color='red', label='Transient')
    plt.scatter(umap_projection[is_event_labels == False, 0], umap_projection[is_event_labels == False, 1], color='blue', label='Background')
    plt.title('UMAP Projection')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend()
    
    if save:
        # Save the plot
        plt.savefig(plot_name)
        plt.close()
    else:
        plt.show()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Load FITS file and create UMAP projection plot.')
    parser.add_argument('file_name', type=str, help='Name of the FITS file to load')
    parser.add_argument('plot_name', type=str, help='Name of the resulting plot')
    args = parser.parse_args()

    plot_umap(args.file_name, args.plot_name)