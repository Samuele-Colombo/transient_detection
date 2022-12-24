# main_parser.py
"""Command-line argument parsing and configuration file reading module.

This module contains a custom argparse action to normalize the values of a tuple, and a function to parse command-line
arguments, read the config file specified by the `--config_file` argument, verify the validity of the input data, and
return a dictionary containing the configuration parameters.

Attributes
----------
NormalizeTupleAction : class
    Custom argparse action to normalize the values of a tuple.

parse : function
    Parses command-line arguments and the config file.
"""

import os.path as osp
import argparse
import torch

from ConfigHandler import read_config

class NormalizeTupleAction(argparse.Action):
    """
    Custom argparse action to normalize the values of a tuple.

    This action checks that all the values in the tuple are strictly positive, and if not, raises a `ValueError` with an
    appropriate error message. If all the values are strictly positive, they are normalized and stored in the namespace.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The ArgumentParser object. Unused in this implementation.
    namespace : argparse.Namespace
        The namespace object to store the normalized values in.
    values : tuple
        The tuple of values to normalize.
    option_string : str, optional
        The option string that specified the argument. Unused in this implementation.

    Raises
    ------
    ValueError
        If any of the values in the tuple is not strictly positive.

    Example
    -------
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--split-fracs', nargs=3, type=float, action=NormalizeTupleAction,
    >>>                     help='Three numbers that represent the fractions of the dataset to use for training, '
    >>>                          'validation, and test sets.')
    >>> args = parser.parse_args(['--split-fracs', '0.6', '0.2', '0.2'])
    >>> print(args.split_fracs)
    (0.4, 0.133, 0.133)
    """
    def __call__(self, parser, namespace, values, option_string=None):
        if any(value <= 0 for value in values):
            raise ValueError('All values in the tuple must be strictly positive.')
        sum_values = sum(values)
        normalized_values = [value / sum_values for value in values]
        setattr(namespace, self.dest, tuple(normalized_values))


def parse():
    """Parses command-line arguments and the config file.
    
    Parameters
    ----------
    --config_file : str
        Path to an INI file storing all the necessary configurations.
    --distributed_init_method : str
        Method for initializing the distributed training setup.
    --distributed_rank : int
        Rank of the current machine in the distributed training setup.
    
    Returns
    -------
    config_args : dict
        A dictionary containing the configuration parameters.
    
    Raises
    ------
    NotADirectoryError
        If any of the directories specified in the config file do not exist or are not directories.
    OSError
        If the config file specified by the `--config_file` argument does not exist or cannot be read.
    AssertionError
        If the values of `k_neighbors` or `learning_rate` in the config file are not greater than zero.
    AssertionError
        If the values of `batch_size`, `num_epochs`, `num_hidden_channels`, or `num_layers` in the config file are not greater than zero.
    RuntimeError
        If the value of the `device` parameter in the config file is not a valid device name.
    """


    parser = argparse.ArgumentParser(description="Optional Arguments")

    parser.add_argument("--config_file", type=str, default='config.ini', required=True, 
                        help="Path to an INI file storing all the necessary configurations.")

    # Add the --distributed_init_method argument
    parser.add_argument("--distributed_init_method", type=str, default="tcp://127.0.0.1:23456", required=True, 
                        help="Method for initializing the distributed training setup")

    # Add the --distributed_rank argument
    parser.add_argument("--distributed_rank", type=int, default=0, required=True, 
                        help="Rank of the current machine in the distributed training setup")

    # Read user input
    parsed_args = parser.parse_args()

    # Read the config file
    config = read_config(parsed_args.config_file)
    config_args = {s : dict(config.items(s)) for s in config.sections()}
    for key in vars(parsed_args):
        config_args["model"][key] = getattr(parsed_args, key)

    # Verify validity of user input
    for name, path in {"root-dir": config_args["data"]["root_dir"],
                       "raw-dir" : osp.join(config_args["data"]["root_dir"], config_args["data"]["raw_dir"]),
                       "processed-dir" : osp.join(config_args["data"]["root_dir"], config_args["data"]["processed_dir"])
                      }.items():
        if not osp.isdir(path):
            if osp.isfile(path):
                raise NotADirectoryError(f"Error: invalid '{name}'. '{path}' refers to a file, not a directory")
            raise NotADirectoryError(f"Error: invalid '{name}'. '{path}' does not exist")
    
    config_args["data"]["k_neighbors"] = int(config_args["data"]["k_neighbors"])
    assert config_args["data"]["k_neighbors"] > 0 , f"Error: invalid 'k_neighbors'. Its value must be > 0, got '{config_args['data']['k_neighbors']}'"
    
    config_args["model"]["learning_rate"] = float(config_args["model"]["learning_rate"])
    assert config_args["model"]["learning_rate"] > 0 , f"Error: invalid 'learning_rate'. Its value must be > 0, got '{config_args['model']['learning_rate']}'"

    names = ["batch_size", "num_epochs", "num_hidden_channels", "num_layers"]
    for name in names:
        value = config_args["model"][name] = int(config_args["model"][name])
        assert value > 0 , f"Error: invalid '{name}'. Its value must be > 0, got '{value}'"

    try:
        # if device is not valid, this raises a RuntimeError
        torch.device(config_args.device)
    except RuntimeError as e:
        raise RuntimeError("Given device name is invalid, `torch` raises the following error:\n"+e.message)

    return config_args
