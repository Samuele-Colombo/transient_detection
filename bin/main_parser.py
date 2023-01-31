# main_parser.py
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

"""Command-line argument parsing and configuration file reading module.

This module contains functions to parse command-line arguments, read the config file specified by the `--config_file` argument, 
verify the validity of the input data, and return a dictionary containing the configuration parameters. It also contains a 
function to normalize the values of a tuple.

Functions
---------
normalize_tuple : function
    Normalizes the values of a tuple.

parse : function
    Parses command-line arguments and the config file.
"""

import os
import os.path as osp
import argparse

from transient_detection.DeepLearning.fileio import bool_flag
from ConfigHandler import read_config

def normalize_tuple(values):
    """Normalizes the values of a tuple.
    
    Parameters
    ----------
    values : tuple
        Tuple of values to be normalized.
    
    Returns
    -------
    normalized_values : tuple
        Normalized tuple of values.
        
    Raises
    ------
    ValueError
        If any of the values in the tuple are not strictly positive.
    """
    if any(value <= 0 for value in values):
        raise ValueError('All values in the tuple must be strictly positive.')
    sum_values = sum(values)
    normalized_values = tuple(value / sum_values for value in values)
    return normalized_values


def parse():
    """
    Parses command-line arguments and an INI configuration file.
    
    This function reads in the following command-line arguments:
    
    --config_file : str
        Required path to an INI file storing all the necessary configurations.
    --distributed_init_method : str
        Method for initializing the distributed training setup. Default is 'tcp://127.0.0.1:23456'.
    --world_size : int
        Number of processes in the distributed training setup. Default is 1.
    --num_workers : int
        Number of workers to be used in distributed training. Default is 0.
    --fast : flag
        If present, uses whatever data is stored in the `processed_dir`, without processing more from the raw data.
    
    It converts the values in the INI file to their correct types, and checks for sensible values.
    It returns a dictionary containing the configuration parameters as specified in the INI file.
    
    Returns
    -------
    config_args : dict
        A dictionary containing the configuration parameters from the INI file and the commandline.
    """


    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to an INI file storing all the necessary configurations.')
    parser.add_argument('--distributed_init_method', type=str, default='tcp://127.0.0.1:23456',
                        help='Method for initializing the distributed training setup.')
    parser.add_argument('--dist_backend', type=str, choices=["nccl", "gloo", "mpi"], default='nccl',
                        help='Distributed training backend. If in doubt consult https://pytorch.org/docs/stable/distributed.html')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of processes in the distributed training setup.')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers in the training loaders.')
    parser.add_argument('--fast', action='store_true',
                        help='If present, uses whatever data is stored in the `processed_dir`, without processing more from the raw data.')
    parser.add_argument('--check_compliance', action='store_true',
                        help='If present, checks whether given files are readable and contain given flags. If not, they are listed in the "compliance_file", if present in the config file.')
    args = parser.parse_args()

    # Check that the specified config file exists
    if not osp.isfile(args.config_file):
        raise OSError(f'The specified config file does not exist: {args.config_file}')

    # Check sanity of world_size
    if args.world_size <= 0:
        raise ValueError(f'world_size value must be positive, {args.world_size} provided')

    # Check sanity of num_workers
    if args.num_workers < 0:
        raise ValueError(f'num_workers value must be zero or positive, {args.num_workers} provided')

    # Read the config file
    config = read_config(args.config_file)

    # Convert the ConfigFile object to a more handy dictionary
    config = {sect: dict(config.items(sect)) for sect in config.sections()}

    # Check sanity of values in the PATHS section. Create missing dirs
    for key, value in config['PATHS'].items():
        if key in ['processed_compacted_out']:
            os.makedirs(osp.dirname(value), exist_ok=True)
        if key in ['processed_data', 'out']:
            os.makedirs(value, exist_ok=True)
        if key in ['data', 'processed_data', 'out']:
            if not osp.exists(value):
                raise OSError(f'{key} does not exist: {value}')

    # Depending on check_compliance flag, check sanity of compliance_file
    if args.check_compliance:
        if "compliance_file" not in config["PATHS"]:
            raise argparse.ArgumentError("'check_compliance' is set to 'True', but no 'compliance_file' provided.")
        os.makedirs(osp.dirname(config["PATHS"]["compliance_file"]), exist_ok=True)
    elif "compliance_file" not in config["PATHS"]:
        config["PATHS"]["compliance_file"] = None
    elif not osp.isfile(config["PATHS"]["compliance_file"]):
        raise OSError(f'The specified compliance file does not exist: {config["PATHS"]["compliance_file"]}')
        

    
    # Convert values in the GENERAL section to their correct types and check their sanity
    for key, value in config['GENERAL'].items():
        if key in ['reset', 'tb']:
            config['GENERAL'][key] = bool_flag(value)
        if key in ['k_neighbors']:
            value = int(value)
            if value <= 0:
                raise ValueError(f'{key} value must be positive, {value} provided')
            config['GENERAL'][key] = value

    # Convert values in the Model section to their correct types and check their sanity
    for key, value in config['Model'].items():
        if key in ['num_layers', 'hidden_dim']:
            value = int(value)
            if value <= 0:
                raise ValueError(f'{key} value must be positive, {value} provided')
            config['Model'][key] = value

    #Convert values in the Dataset section to their correct types and check their sanity
    for key, value in config['Dataset'].items():
        if key in ['batch_per_gpu']:
            value = int(value)
            if value <= 0:
                raise ValueError(f'{key} value must be positive, {value} provided')
            config['Dataset'][key] = value
        if key in ['shuffle']:
            config['Dataset'][key] = bool_flag(value)

    #Convert values in the Trainer section to their correct types and check their sanity
    for key, value in config['Trainer'].items():
        if key in ['epochs', 'save_every']:
            value = int(value)
            if value <= 0:
                raise ValueError(f'{key} value must be positive, {value} provided')
            config['Trainer'][key] = value
        if key in ['fp16']:
            config['Trainer'][key] = bool_flag(value)

    #Convert values in the Optimization section to their correct types and check their sanity
    for key, value in config['Optimization'].items():
        if key in ['lr_start','lr_end','lr_warmup']:
            config['Optimization'][key] = float(value)
            
    # convert split_fracs to correct types and check their sanity
    split_fracs = tuple(map(float, config['Dataset']['split_fracs'].split(',')))
    config['Dataset']['split_fracs'] = normalize_tuple(split_fracs)
    if not len(split_fracs) == 3:
        raise ValueError(f'Invalid number of split fractions: {len(split_fracs)}. Must be 3.')
    
    # convert keys in keys list
    config['Dataset']['keys'] = list(map(lambda key: key.strip().strip('\"\''), config['Dataset']['keys'].split(',')))
        
    #Merge commandline arguments with INI file configs
    config_args = {**vars(args), **config}
    return config_args

