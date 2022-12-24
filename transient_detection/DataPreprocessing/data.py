# DataPreprocessing/data.py
"""
Defines datatypes for preprocessing and data learning.

This module defines the `IcaroData` and `IcaroDataset` classes, which are used to read and process raw data and store
it in `torch_geometric.data.Data` objects. The `IcaroData` class is a subclass of `torch_geometric.data.Data` that
adds a `pos` property to store and retrieve the position data. The `IcaroDataset` class reads and processes raw data
from the specified directories, and stores the processed data in `IcaroData` objects. The `IcaroDataset` class also
provides methods for retrieving the raw and processed file names, and for processing the data in parallel using either
threads or separate processes.

Examples
--------
>>> import os
>>> from transient_detection.DataPreprocessing.data import IcaroDataset
>>> 
>>> raw_dir = os.path.join("path", "to", "raw", "data")
>>> processed_dir = os.path.join("path", "to", "processed", "data")
>>> genuine_pattern = "genuine*.npy"
>>> simulated_pattern = "simulated*.npy"
>>> keys = ["timestamps", "signals", "rfi"]
>>> 
>>> dataset = IcaroDataset(raw_dir, genuine_pattern, simulated_pattern, keys,
>>>                       raw_dir=raw_dir, processed_dir=processed_dir)
>>> dataset.process()
>>> 
>>> for data in dataset:
>>>     print(data)

Notes
-----
The `read_events()` function is imported from the `utilities` module.

"""

import os
import os.path as osp
import concurrent.futures
from glob import glob

import numpy as np
import torch

import pyg_lib #new in torch_geometric 2.2.0!
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from sklearn.preprocessing import StandardScaler

from transient_detection.DataPreprocessing.utilities import read_events



class IcaroData(Data):
    """
    Subclass of `torch_geometric.data.Data` that adds a `pos` property to store and retrieve the position data. The position data is stored in the last three values of the `x` attribute.
    
    Parameters
    ----------
    None
    
    Attributes
    ----------
    pos : np.ndarray
        3-D position data.
    """
    def __init__(self, x = None, edge_index = None, edge_attr = None, y = None, pos = None, **kwargs):
        assert pos is None, ("This subclass of `Data` reimplemnts the `pos` property so that it corresponds to the last three"+ 
                             " values of the `x` attribute. Please append the position coordinates to your 'x' parameter")
        assert x.shape[1] >= 3, ("This subclass of `Data` reimplemnts the `pos` property so that it corresponds to the last three"+ 
                                 " values of the `x` attribute. Therefore the 'x' parameter must contain at least three elements.")
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)

    @property
    def pos(self):
        """
        Getter for the position data.
        
        Returns
        -------
        np.ndarray
            3-D position data.
        """
        return self.x[:, -3:]

    @pos.setter
    def pos(self, replace):
        """
        Setter for the position data.
        
        Parameters
        ----------
        replace : np.ndarray
            3-D position data to be set.
        
        Raises
        ------
        AssertionError
            If the shape of `replace` is not (num_points, 3).
        """
        assert replace.shape == self.pos.shape
        self.x[:, -3:] = replace


class IcaroDataset(Dataset):
    def __init__(self,
                 root,
                 genuine_pattern:str,
                 simulated_pattern:str,
                 keys,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 raw_dir=None,
                 processed_dir=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initializes an `IcaroDataset` object.
        
        Parameters
        ----------
        root : str
            Root directory of the dataset.
        genuine_pattern : str
            Glob pattern to match genuine data files.
        simulated_pattern : str
            Glob pattern to match simulated data files.
        keys : list of str
            List of keys to extract from the data files.
        transform : callable, optional
            Transform to apply to the data.
        pre_transform : callable, optional
            Pre-processing transform to apply to the data.
        pre_filter : callable, optional
            Pre-processing filter to apply to the data.
        raw_dir : str, optional
            Directory containing the raw data files.
        processed_dir : str, optional
            Directory to store the processed data files.
        device : torch.device, optional
            Device to store the data on.
        """
        self._raw_dir          = raw_dir
        self._processed_dir    = processed_dir
        self.genuine_pattern   = genuine_pattern
        self.simulated_pattern = simulated_pattern
        self.keys              = keys
        self.device            = device
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self):
        """
        Processes a raw data file.
        
        Parameters
        ----------
        raw_path : str
            Path to the raw data file.
        
        Returns
        -------
        IcaroData
            Processed data.
        """
        if osp.isabs(self._raw_dir):
            return self._raw_dir
        return osp.join(os.getcwd(), self.raw_dir)

    @property
    def processed_dir(self):
        if osp.isabs(self._processed_dir):
            return self._processed_dir
        return osp.join(os.getcwd(), self._processed_dir)

    @property
    def raw_file_names(self):
        return list(sorted(list(glob(osp.join(self.raw_dir, self.genuine_pattern)) + 
                                glob(osp.join(self.raw_dir, self.simulated_pattern))
                    )))

    @property
    def processed_file_names(self):
        return list(map(lambda name: osp.join(self.processed_dir, osp.basename(name)+".pt"),
                        glob(osp.join(self.raw_dir, self.simulated_pattern))))

    @property
    def num_classes(self):
        return 2

    def _hidden_process(self, raw_path):
        """
        Processes a raw data file and returns the resulting data as an `IcaroData` object.
        
        The raw data file is read using the `read_events` function from the `utilities` module. The data is then transformed into an `IcaroData` object by storing the data in the appropriate attributes (`x` and `y`) and normalizing the position data using a `StandardScaler` from scikit-learn. Finally, the `IcaroData` object is moved to the device specified by the `device` attribute.
        
        Parameters
        ----------
        raw_path : str
            Path to the raw data file.
        
        Returns
        -------
        IcaroData
            Processed data.
        """
        dat = read_events(*raw_path)
        data = IcaroData(x = torch.from_numpy(np.array([dat[key] for key in self.keys]).T).float(),
                         y = torch.from_numpy(np.array(dat["ISFAKE"])).long())

        ss2 = StandardScaler()
        ss2.fit(data.pos)
        new_pos = ss2.transform(data.pos)
        data.pos = torch.tensor(new_pos)
        data.to(self.device)

        if self.pre_filter is not None and not self.pre_filter(data):
            return

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, osp.join(self.processed_dir, osp.basename(raw_path[-1])+".pt"))
        del data


    def process(self):
        """
        Processes a raw data file and stores the resulting data in a file.
        
        Parameters
        ----------
        raw_path : str
            Path to the raw data file.
        
        Returns
        -------
        str
            Path to the processed data file.
        """
        already_processed = os.listdir(self.processed_dir)
        fnames = ((genuine, simulated) for genuine, simulated in zip(
                        sorted(glob(osp.join(self.raw_dir, self.genuine_pattern))),
                        sorted(glob(osp.join(self.raw_dir, self.simulated_pattern)))
                    ) if osp.basename + ".pt" in already_processed
                 )
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(self._hidden_process, fnames)

    def len(self):
        """
        Returns the number of data points in the dataset.
        
        Returns
        -------
        int
            Number of data points in the dataset.
        """
        return len(self.processed_file_names)

    def get(self, idx):
        """
        Returns the data point at the specified index.
        
        Parameters
        ----------
        idx : int
            Index of the data point to retrieve.
        
        Returns
        -------
        IcaroData
            Data point at the specified index.
        """
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data
