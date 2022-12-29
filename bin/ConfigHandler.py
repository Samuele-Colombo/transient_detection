# ConfigHandler.py
"""Configuration file handling module.

This module contains functions to read and create configuration files in INI format. It includes a function to read a
configuration file and return a configuration object, a function to create a default configuration file with the given
filename, and a function to get a default configuration object.

Attributes
----------
read_config : function
    Read a configuration file and return a configuration object.
create_default_config : function
    Create a default configuration file with the given filename.
get_default_config : function
    Get a default configuration object.
"""

import os
import os.path as osp
import configparser

def read_config(filename: str) -> configparser.ConfigParser:
    """
    Read a configuration file and return a configuration object.
    
    This function reads an INI configuration file and returns a configuration object. If the
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
        raise Exception(f"Error: we find present in file but not needed '{repr(rd_diff)}' and needed but not present in file '{repr(dr_diff)}'")
    
    return config


def create_default_config(filename: str) -> None:
    """
    Create a default configuration file with the given filename.
    
    This function retrieves the default configuration object and writes it to the specified file.
    
    Parameters
    ----------
    filename : str
        The name of the configuration file to create.
    
    """


    config = get_default_config()

    # Write the config file
    with open(filename, 'w') as configfile:
        config.write(configfile)

def get_default_config() -> configparser.ConfigParser:
    """
    Get a default configuration object.
    
    This function creates a default configuration object with two sections: 'data' and 'model'.
    The 'data' section contains settings related to data processing and loading, including the root
    directory, directories for raw and processed data, and patterns for genuine and simulated data
    files. The 'model' section contains settings related to model training and evaluation, including
    the learning rate, device name, batch size, number of epochs, number of hidden channels, number of
    layers, and split fractions for the data. The configuration object also includes comments for each
    section and key-value pair.
    
    Returns
    -------
    configparser.ConfigParser
        The default configuration object.
    
    """

    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Set default values for the config file
    config['data'] = {
        'root_dir': os.getcwd(),  # The current working directory
        'raw_dir': 'raw',  # Directory for raw data
        'processed_dir': 'processed',  # Directory for processed data
        'genuine_pattern': '*EVLI0000.FTZ',  # Pattern for genuine data files
        'simulated_pattern': '*EVLF0000.FTZ',  # Pattern for simulated data files
        'k_neighbors': '',  # Number of neighbors to consider in k-NN algorithm
    }

    config['model'] = {
        'learning_rate': '',  # Learning rate for the model
        'weight_decay': '5e-4',  # Weight decay for the model
        'device_name': 'cuda',  # Device to use for training
        'batch_size': '128',  # Batch size for training
        'num_epochs': '100',  # Number of epochs to train for
        'num_hidden_channels': '',  # Number of hidden channels in the model
        'num_layers': '',  # Number of layers in the model
        'split_fracs': '0.6,0.2,0.2',  # Split fractions for the data
    }

    # Add comments to the config file
    config.add_comment('data', 'The data section contains settings related to data processing and loading')
    config.add_comment('data', 'root_dir', 'The root directory for the data')
    config.add_comment('data', 'raw_dir', 'The directory for raw data, relative to `root_dir`')
    config.add_comment('data', 'processed_dir', 'The directory for processed data, relative to `root_dir`')
    config.add_comment('data', 'genuine_pattern', 'The pattern for genuine data files')
    config.add_comment('data', 'simulated_pattern', 'The pattern for simulated data files')
    config.add_comment('data', 'k_neighbors', 'The number of neighbors to consider in k-NN algorithm')

    config.add_comment('model', 'The model section contains settings related to model training and evaluation')
    config.add_comment('model', 'learning_rate', 'The learning rate for the model')
    config.add_comment('model', 'weight_decay', 'The weight decay for the model')
    config.add_comment('model', 'device_name', 'The device to use for training')
    config.add_comment('model', 'batch_size', 'The batch size for training')
    config.add_comment('model', 'num_epochs', 'The number of epochs to train for')
    config.add_comment('model', 'num_hidden_channels', 'The number of hidden channels in the model')
    config.add_comment('model', 'num_layers', 'The number of layers in the model')
    config.add_comment('model', 'split_fracs', 'The split fractions for the dataset')

    return config