# DataPreprocessing/data.py
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

"""
Defines datatypes for preprocessing and data learning.

This module defines the `SimTransientData` and `SimTransientDataset` classes, which are used to read and process raw data and store
it in `torch_geometric.data.Data` objects. The `SimTransientData` class is a subclass of `torch_geometric.data.Data` that
adds a `pos` property to store and retrieve the position data. The `SimTransientDataset` class reads and processes raw data
from the specified directories, and stores the processed data in `SimTransientData` objects. The `SimTransientDataset` class also
provides methods for retrieving the raw and processed file names, and for processing the data in parallel using either
threads or separate processes.

Examples
--------
>>> import os
>>> from transient_detection.DataPreprocessing.data import SimTransientDataset
>>> 
>>> raw_dir = os.path.join("path", "to", "raw", "data")
>>> processed_dir = os.path.join("path", "to", "processed", "data")
>>> genuine_pattern = "genuine*.npy"
>>> simulated_pattern = "simulated*.npy"
>>> keys = ["timestamps", "signals", "rfi"]
>>> 
>>> dataset = SimTransientDataset(raw_dir, genuine_pattern, simulated_pattern, keys,
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
import fnmatch
from glob import glob
import logging

import numpy as np
import torch

from astropy.table.np_utils import TableMergeError

# import pyg_lib #new in torch_geometric 2.2.0!
from torch_geometric.data import Data
from torch_geometric.data import Dataset

from transient_detection.DataPreprocessing.utilities import read_events, get_paired_filenames, in2d, get_uncompliant
from transient_detection.DataPreprocessing.utilities import StandardScaler



class SimTransientData(Data):
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


class SimTransientDataset(Dataset):
    def __init__(self,
                 genuine_pattern:str,
                 simulated_pattern:str,
                 keys,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 raw_dir=None,
                 processed_dir=None,
                 rank=0,
                 world_size=1,
                 compliance_file=None):
        """
        Initializes an `SimTransientDataset` object.
        
        Parameters
        ----------
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
        """
        if raw_dir == processed_dir:
            raise RuntimeError(f"'raw_dir' and 'processed_dir' refer to the same path: {raw_dir}")
        if genuine_pattern == simulated_pattern:
            raise RuntimeError(f"'genuine_pattern' and 'simulated_pattern' refer to the same pattern: {genuine_pattern}")
        self._raw_dir          = raw_dir
        self._processed_dir    = processed_dir
        self.genuine_pattern   = genuine_pattern
        self.simulated_pattern = simulated_pattern
        self.keys              = keys
        self.rank              = rank
        self.world_size        = world_size
        self.compliance_file   = compliance_file
        super().__init__(osp.commonpath([raw_dir, processed_dir]), transform, pre_transform, pre_filter)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    def _slice(self, length: int) -> slice:
        """
        Return a slice object for the current rank given a total length.
            
        Parameters
        ----------
        length : int
            Total length of the object to be sliced

        Returns
        -------
        slice
            the `self.rank`th `slice` out of `self.world_size` of the [0, `length`) range.
        """
        return slice((self.rank*length)//self.world_size, 
                     ((self.rank+1)*length)//self.world_size
                    )

    @property
    def raw_file_names(self):
        rfn_list = sum(map(list, get_paired_filenames(self.raw_dir, self.genuine_pattern, self.simulated_pattern)))
        return rfn_list[self._slice(len(rfn_list))]

    @property
    def processed_file_names(self):
        pfn_list = list(map(lambda names: osp.join(self.processed_dir, osp.basename(names[-1])+".pt"), 
                            get_paired_filenames(self.raw_dir, self.genuine_pattern, self.simulated_pattern)
                    ))
        self._len = len(pfn_list)
        return pfn_list[self._slice(len(pfn_list))]

    @property
    def num_classes(self):
        return 2

    @torch.no_grad()
    def _hidden_process(self, filenames):
        try:
            dat = read_events(*filenames, keys=self.keys)
        except TableMergeError as e:
            print(f"Ignoring files {filenames}")
            logging.exception("Raised error is:")
            if self.compliance_file is not None:
                print("Listing them as uncompliant in 'compliance_file'")
                with open(self.compliance_file, 'a') as f:
                    f.write(" ".join(filenames)+'\n')
            return
        except Exception as e:
            print(f"Got error while reading from files {filenames}.")
            raise e
        data = SimTransientData(x = torch.from_numpy(np.array([dat[key] for key in self.keys]).T).float(),
                                y = torch.from_numpy(np.array(dat["ISSIMULATED"])).long()).cuda()

        ss2 = StandardScaler()
        ss2.fit(data.pos)
        new_pos = ss2.transform(data.pos)
        data.pos = torch.tensor(new_pos, device=new_pos.device())

        if self.pre_filter is not None and not self.pre_filter(data):
            return

        if self.pre_transform is not None:
            try:
                data = self.pre_transform(data)
            except Exception as e:
                print(f"Got error while pre-transforming from files {filenames}.")
                raise e

        torch.save(data, osp.join(self.processed_dir, osp.basename(filenames[-1])+".pt"))
        del data


    def process(self):
        already_processed = np.array(os.listdir(self.processed_dir))
        uncompliant_pairs = np.array(list(get_uncompliant(self.compliance_file)))
        gsnames = np.array(list(get_paired_filenames(self.raw_dir, self.genuine_pattern, self.simulated_pattern)))
        gsnames = gsnames[np.logical_not(in2d(gsnames, uncompliant_pairs))]
        gsbasenames=np.vectorize(osp.basename)(gsnames.T[0])
        gsnames = gsnames[np.logical_not(np.in1d(np.char.add(gsbasenames, ".pt"), already_processed))]
        assert gsnames.shape[-1] == 2
        np.apply_along_axis(self._hidden_process, -1, gsnames)


    def len(self):
        """
        Returns the number of data points in the dataset.
        
        Returns
        -------
        int
            Number of data points in the dataset.
        """
        if not hasattr(self, "_len"):
            self._len = len(self.processed_file_names)
        return self._len

    def get(self, idx):
        """
        Returns the data point at the specified index.
        
        Parameters
        ----------
        idx : int
            Index of the data point to retrieve.
        
        Returns
        -------
        SimTransientData
            Data point at the specified index.
        """
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data

class FastSimTransientDataset(Dataset):
    """
    A dataset for loading and interacting with data stored in PyTorch files.

    Parameters
    ----------
    root : str
        The root directory of the dataset.
    pattern : str, optional
        A glob pattern to match file names. Defaults to '*EVLF000.FTZ*'.
    transform : callable, optional
        A function/transform to apply to the data. Defaults to None.
    """
    def __init__(self, root, pattern='*EVLF000.FTZ*', transform=None):
        super().__init__(root=root, transform=transform)
        self._raw_dir = root
        search_path = os.path.join(root, pattern)
        self.filenames = sorted(glob(search_path))
        self.file_count = len(self.filenames)

    @property
    def raw_dir(self):
        """The root directory of the raw data."""
        return self._raw_dir

    @property
    def processed_dir(self):
        """The root directory of the processed data."""
        return self._raw_dir

    @property
    def raw_file_names(self):
        """The names of the raw files in the dataset."""
        return self.filenames

    @property
    def processed_file_names(self):
        """The names of the processed files in the dataset."""
        return self.filenames

    def len(self):
        """The number of files in the dataset."""
        return self.file_count

    def get(self, idx):
        """
        Get a data item from the dataset by its index.

        Parameters
        ----------
        idx : int
            The index of the data item.

        Returns
        -------
        object
            The data item at the given index.
        """
        data = torch.load(self.filenames[idx])
        return data
