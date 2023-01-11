# ConfigHandler.py
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

"""Configuration file handling module.

This module contains functions to read and create INI format configuration files. It includes a function to read a configuration file and return a ConfigParser object, a function to create a default configuration file with the given filename, and a function to get a default ConfigParser object.

Attributes
----------
read_config : function
    Read a configuration file and return a ConfigParser object.
create_default_config : function
    Create a default configuration file with the given filename.
get_default_config : function
    Get a default ConfigParser object.
"""

import os.path as osp
import configparser

def read_config(filename: str) -> configparser.ConfigParser:
    """
    Read a configuration file and return a ConfigParser object.

    This function reads an INI configuration file and returns a ConfigParser object. If the
    specified file does not exist, it is created with default values. If the file is incompatible
    with the expected structure, an exception is raised.

    Parameters
    ----------
    filename : str
        The name of the configuration file to read.

    Returns
    -------
    configparser.ConfigParser
        The configuration object.

    Raises
    ------
    Exception
        If the specified file does not end with the '.ini' suffix.
    IsADirectoryError
        If the specified file points to a directory.
    Exception
        If the specified file does not exist and is created with default values.
    Exception
        If the specified file is incompatible with the expected structure.

    Examples
    --------
    >>> config = read_config('config.ini')
    """
    # Initialize a config parser object
    config = configparser.ConfigParser()

    # Check that the file has the correct suffix
    if not filename.endswith(".ini"): 
        raise Exception(f"Error: `filename` must end with the '.ini' suffix, got '{filename}'.")

    # Check that the file is not a directory
    if osp.isdir(filename):
        raise IsADirectoryError(f"Error: `filename` must not point to a directory, '{filename}' is a directory.")

    # If the file does not exist, create it with default values
    if not osp.exists(filename):
        create_default_config(filename=filename)
        raise Exception(f"Argument `filename` did not point to an existing config file, a new one was created containing default values. Modify it opportunely.")

    # Read the config file
    config.read(filename)

    # Ensure the config file is compatible with requirements
    def_config = get_default_config()
    read_set = {s + '|' + k for s in config.sections() for k, _ in config.items(s)}
    def_set = {s + '|' + k for s in def_config.sections() for k, _ in def_config.items(s)}
    rd_diff = read_set - def_set
    dr_diff = def_set - read_set

    if rd_diff.union(dr_diff):
        raise Exception(f"Error: we find present in file but not needed '{repr(rd_diff)}' and needed but not present in file '{repr(dr_diff)}'.")

    return config

def get_default_config_text() -> str:
    return """PATHS:
        data: '/path/to/raw/data'  # Path to raw dataset directory
        processed_data: '/path/to/processed/data'  # Path to processed dataset directory
        processed_pattern: '*EVLF0000.FTZ.pt'  # Pattern for processed data files
        genuine_pattern: '*EVLI0000.FTZ'  # Pattern for genuine data files
        simulated_pattern: '*EVLF0000.FTZ'  # Pattern for simulated data files
        out: 'out'  # Path to out directory
    GENERAL:
        reset: false  # Reset saved model logs and weights
        tb: true  # Start TensorBoard
        k_neighbors: 6  # Number of neighbors to consider in k-NN algorithm
    Model:
        model: 'gcn'  # Model name
        num_layers: 2  # Number of layers
        hidden_dim: 4  # Number of nodes in the hidden layer.
    Dataset:
        batch_per_gpu: 96  # Batch size per gpu
        shuffle: true  # Shuffle dataset
        # Columns to be used as data features, last three must be the position features
        keys:  
        - these
        - are
        - the
        - keys
        # Fractions to split the dataset into. Normalization not necessary
        split_fracs: 
        - 0.6 # Training split
        - 0.2 # Validation split
        - 0.2 # Testing split
    Trainer:
        epochs: 1000  # Number of epochs
        save_every: 10  # Save model every n epochs
        fp16: true  # Use fp16
    Optimization:
        optimizer: 'adam'  # Optimizer to choose between 'adam', 'sgd', and 'adagrad'
        lr_start: 0.0005  # Learning rate start
        lr_end: 1e-06  # Learning rate end
        lr_warmup: 10  # Learning rate warmup
    """

def create_default_config(filename: str) -> None:
    """
    Create a default configuration file with the given filename.
    This function creates a default INI configuration file with the given filename if it does not exist.
    If the file already exists, it is overwritten.

    Parameters
    ----------
    filename : str
        The name of the configuration file to be created.

    Returns
    -------
    None

    Raises
    ------
    Exception
        If the specified file does not end with the '.ini' suffix.
    IsADirectoryError
        If the specified file points to a directory.

    Examples
    --------
    >>> create_default_config('config.ini')
    """
    if not filename.endswith(".ini"): 
        raise Exception(f"Error: `filename` must end with the '.ini' suffix, got '{filename}'.")

    if osp.isdir(filename):
        raise IsADirectoryError(f"Error: `filename` must not point to a directory, '{filename}' is a directory.")

    with open(filename, "w") as f:
        f.write(get_default_config_text())
def get_default_config() -> configparser.ConfigParser:
    """
    Get a default configuration object.
    
    Returns
    -------
    configparser.ConfigParser
        The default configuration object.
    """
    result = get_default_config_text()

    # Create a ConfigParser object
    config = configparser.RawConfigParser(allow_no_value=True)
    config.read_string(result)

    return config
