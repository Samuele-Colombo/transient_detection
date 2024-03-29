#!python
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
# Graph Neural Network for Simulated X-Ray Transient Detection

The present work aims to train a GNN to label a particular sort of X-Ray transient using simulated events 
overlayed onto real data from XMM-Newton observations. We will experiment with Graph Convolutional Networks (GCNs).
We will therefore  have to trandsform our point-cloud data into a "k nearest neighbors"-type graph. 
"""

import os
import os.path as osp
import functools
import datetime

import torch
from torch.utils.data import random_split
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import torch_geometric
import torch_geometric.transforms as ttr
from torch_geometric.loader import DataLoader

from transient_detection.DataPreprocessing.data import FastSimTransientDataset, SimTransientDataset
from transient_detection.DeepLearning.models import GCNClassifier
from transient_detection.DeepLearning.utilities import loss_func, print_with_rank_index
from transient_detection.DeepLearning.optimizers import get_optimizer
from transient_detection.DeepLearning.distributed import fix_random_seeds
from transient_detection.DeepLearning.trainer import Trainer
from transient_detection.DeepLearning import fileio

from transient_detection.main_parser import parse
from ancillary_scripts.check_compliance import check_compliance

def main():

    ngpus_per_node = torch.cuda.device_count()
    
    """ This next line is the key to getting DistributedDataParallel working on SLURM:
    SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
    current process inside a node and is also 0 or 1 in this example."""

    local_rank = int(os.environ.get("SLURM_LOCALID")) if os.environ.get("SLURM_LOCALID") else 0 
    rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank if os.environ.get("SLURM_NODEID") else 0

    # Override the built-in print function with the custom function
    print = functools.partial(print_with_rank_index, rank)
    
    print("parsing arguments")
    args = parse()

    world_size = args["world_size"]

    current_device = local_rank
    print("Name of device: ", torch.cuda.get_device_name(current_device))
    torch.cuda.set_device(current_device)

    """ this block initializes a process group and initiate communications
		between all processes running on all nodes """

    print('Initializing Process Group...')
    #init the process group
    mp.set_start_method("spawn")
    timeout = datetime.timedelta(hours=1)
    dist.init_process_group(backend=args["dist_backend"], init_method=args["distributed_init_method"], world_size=world_size, rank=rank, timeout=timeout)
    fix_random_seeds()
    ismain = args["main"] = (rank == 0)
    print("process group ready!")

    genuine_pattern   = osp.join("0*","pps","P*EVLI0000.FTZ")
    simulated_pattern = osp.join("0*","pps","P*EVLF0000.FTZ")
    if args["test"]:
        genuine_pattern   = "*.bkg.fits"
        simulated_pattern = "*.evt.fits"
    if args["group"]:
        gendir, genname = osp.split(genuine_pattern)
        prefix = "group" if genname.startswith('*') else "group*"
        genuine_pattern = osp.join(gendir, prefix+genname)
        simdir, simname = osp.split(simulated_pattern)
        prefix = "group" if simname.startswith('*') else "group*"
        simulated_pattern = osp.join(simdir, prefix+simname)

    keys = args["Dataset"]["keys"]
    raw_dir = args["PATHS"]["data"]

    if args["check_compliance"]:
        if ismain:
            print("Checking compliance...")
            import numpy as np
            from transient_detection.DataPreprocessing.utilities import get_paired_filenames
            gsnames = np.array(list(get_paired_filenames(raw_dir, genuine_pattern, simulated_pattern)))
            check_compliance(gsnames, rank, world_size, keys)
        else:
            print("Waiting for compliance check...")
        dist.barrier()

    print('Making dataset..')

    processed_dir = args["PATHS"]["processed_data"]

    print('- Checking if raw_dir has to be extracted')
    if osp.isfile(raw_dir):
        raw_archive = raw_dir
        raw_dir = osp.join(os.environ.get("SLURM_TMPDIR"), "raw_data")
        if ismain:
            print(f"\t- Extracting raw data from {raw_archive}...")
            fileio.extract(raw_archive, raw_dir)
            print("\t- Done!")
        dist.barrier()
    
    print('- Checking if processed_dir has to be extracted')
    if osp.isfile(processed_dir):
        processed_archive = processed_dir
        processed_dir = osp.join(os.environ.get("SLURM_TMPDIR"), "processed_data")
        if ismain:
            print(f"\t- Extracting processed data from {processed_archive}...")
            fileio.extract(processed_archive, processed_dir)
            print("\t- Done!")
        dist.barrier()

    if not args["fast"]:
        print('- Converting data to graphs')
        compliance_file = args["PATHS"]["compliance_file"]
        SimTransientDataset(genuine_pattern = genuine_pattern, 
                            simulated_pattern = simulated_pattern, 
                            raw_dir = raw_dir,
                            processed_dir = processed_dir,
                            keys=keys,
                            pre_transform = ttr.KNNGraph(k=args["GENERAL"]["k_neighbors"], force_undirected=True),
                            rank = rank,
                            world_size = world_size,
                            compliance_file = compliance_file
                           )
        # if ismain and "processed_compacted_out" in args["PATHS"]:
        #     new_processed_archive = args["PATHS"]["processed_compacted_out"]
        #     fileio.compact(processed_dir, new_processed_archive)
        dist.barrier()
        if not os.environ.get("SLURM_LOCALID") is None:
            return
    
    print('- Loading dataseta')
    ds = FastSimTransientDataset(root = processed_dir, 
                                 pattern = osp.basename(simulated_pattern)+".pt",
                                 device="cuda:{}".format(current_device)
                                )

    assert len(ds) > 0, f"FastSimTransientDataset found no processed data in '{processed_dir}' with pattern '{osp.basename(simulated_pattern)+'.pt'}'"

    print('Making model..')

    num_hidden_channels = args["Model"]["hidden_dim"]
    num_layers = args["Model"]["num_layers"]


    model = GCNClassifier(num_layers = num_layers, 
                        input_dim  = ds.num_node_features, 
                        hidden_dim = num_hidden_channels, 
                        output_dim = ds.num_classes if ds.num_classes > 2 else 1
                        )

    num_workers = args["num_workers"]
    batch_size = args["Dataset"]["batch_per_gpu"]
    split_fracs = args["Dataset"]["split_fracs"]

    train_dataset, val_dataset, test_dataset = random_split(ds, split_fracs)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, # num_workers=num_workers, 
                              shuffle=(train_sampler is None), sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)#, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    model.cuda()
    model = DistributedDataParallel(model, device_ids=[current_device], output_device=current_device)

    optimizer = get_optimizer(model=model, args=args)

    print('Training model..')
    
    Trainer(args=args, train_loader=train_loader, validation_loader=val_loader, model=model, loss=loss_func, optimizer=optimizer).fit()

if __name__ == "__main__":
    main()