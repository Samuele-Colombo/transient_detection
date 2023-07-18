from glob import glob
import os.path as osp
import heapq

from transient_detection.DataPreprocessing.data import FastSimTransientDataset
from transient_detection.main_parser import parse
from transient_detection.DeepLearning.models import GCNClassifier
from transient_detection.DeepLearning.utilities import loss_func
import torch

import json
import numpy as np
import pandas as pd
import os.path as osp

def generate_table(filename):
    filedir =  osp.split(osp.split(filename)[0])[1]
    with open(filename, 'r') as file:
        data=pd.DataFrame(json.loads(line) for line in file)
    data["Run"] = np.full_like(data.iloc[:, 1], filedir)
    data["Epoch"] = data["Epoch"].astype(int)
    data["Validation"] = data["Validation"].astype(bool)
    for column in data.columns:
        if column in ["Run", "Epoch", "Validation"]: continue
        data[column] = data[column].apply(lambda x: x.split()[0]).astype(float)
    data.set_index("Run")
    return data

def get_best_worst(args, weightspath, num = 10): 
    json_files = np.array(list(glob(osp.join(args["PATHS"]["out"],"*json"))))
    data = pd.concat(list(generate_table(file) for file in json_files), axis=0)
    data = data.loc[data["Validation"]].drop("Validation", axis=1)
    grouped = data.loc[data['loss'].idxmin()].sort_values('loss')
    grouped.set_index("Run", inplace=True)
    # print(grouped)
    bestindex = grouped.index[0]
    bestepoch = grouped["Epoch"][0]
    grouped = grouped.iloc[0]
    # print(bestindex)
    # print(grouped)

    # args = parse()
    current_device = 0
    raw_dir = args["PATHS"]["data"]
    processed_dir = args["PATHS"]["processed_data"]
    simulated_pattern = osp.join("0*","pps","*EVLF0000.FTZ")
    if args["test"]:
        simulated_pattern = "*.evt.fits"

    ds = FastSimTransientDataset(root = processed_dir, 
                                 pattern = osp.basename(simulated_pattern)+".pt",
                                 device="cuda:{}".format(current_device)
                                )
    
    weightspath = osp.join(args["PATHS"]["out"], "weights", "gcn", "Epoch_{:03}.pth".format(bestepoch))

    ckpt = torch.load(weightspath, map_location='cpu')

    num_hidden_channels = args["Model"]["hidden_dim"]
    num_layers = args["Model"]["num_layers"]

    print('Initializing Process Group...')
    from torch.nn.parallel import DistributedDataParallel
    import torch.multiprocessing as mp
    import torch.distributed as dist
    import datetime
    #init the process group
    mp.set_start_method("spawn")
    timeout = datetime.timedelta(hours=1)
    dist.init_process_group(backend=args["dist_backend"], init_method=args["distributed_init_method"], world_size=1, rank=0, timeout=timeout)

    model = GCNClassifier(num_layers = num_layers, 
                        input_dim  = ds.num_node_features, 
                        hidden_dim = num_hidden_channels, 
                        output_dim = 1
                        )
    model = DistributedDataParallel(model.cuda())#, device_ids=[current_device], output_device=current_device)
    # bestloss = np.inf
    # worstloss = 0
    # bestfile = None
    # worstfile = None

    filesheap = []
    from tqdm import tqdm
    progress = tqdm(total=len(ds))
    for i, obs in enumerate(ds):
        if obs.y.sum() == len(obs.y): continue
        model.load_state_dict(ckpt['model'])
        model.eval()
        input_data = obs.x.cuda(non_blocking=True)
        edge_indices = obs.edge_index.cuda(non_blocking=True)
        edge_attr = obs.edge_attr.cuda(non_blocking=True)
        out = model(input_data, edge_indices, edge_attr)
        loss, *_ = loss_func(out, obs.y)
        # if loss < bestloss:
        #     bestloss = loss
        #     bestfile = osp.join(raw_dir, osp.basename(ds.raw_file_names[i].replace(".pt", "")))
        # elif loss > worstloss:
        #     worstloss = loss
        #     worstfile = osp.join(raw_dir, osp.basename(ds.raw_file_names[i].replace(".pt", "")))
        basename = osp.basename(ds.raw_file_names[i].replace(".pt", ""))
        obsid = basename[1:11]
        heapq.heappush(filesheap, (loss, osp.join(raw_dir, obsid, "pps", basename)))
        progress.update(1)

    best_files = [file for (_, file) in heapq.nsmallest(num, filesheap)]
    worst_files = [file for (_, file) in heapq.nlargest(num, filesheap)]

    return best_files, worst_files

if __name__ == "__main__":
    args = parse()
    best_files, worst_files = get_best_worst(args, 10)

    print("best_files:\n", best_files)
    print()
    print("worst_files:\n", worst_files)

    

    # print(list(generate_table(file) for file in json_files))
