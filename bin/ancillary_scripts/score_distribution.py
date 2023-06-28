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

if __name__ == "__main__":
    from glob import glob
    import os.path as osp
    import heapq

    from transient_detection.DataPreprocessing.data import FastSimTransientDataset
    from transient_detection.main_parser import parse
    from transient_detection.DeepLearning.models import GCNClassifier
    from transient_detection.DeepLearning.utilities import loss_func
    import torch

    json_files = np.array(list(glob(osp.join("test", "Icaro", "out_*","*json"))))
    data = pd.concat(list(generate_table(file) for file in json_files), axis=0)
    data = data.loc[data["Validation"]].drop("Validation", axis=1)
    grouped = data.loc[data['loss'].idxmin()].sort_values('loss')
    grouped.set_index("Run", inplace=True)
    # print(grouped)
    bestindex = grouped.index[0]
    bestepoch = grouped["Epoch"][0]
    grouped = grouped.iloc[0]
    print("best model is: ", bestindex)
    # print(grouped)

    args = parse()
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
    
    weightspath = osp.join("test", "Icaro", bestindex, "weights", "gcn", "Epoch_{:03}.pth".format(bestepoch))

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

    scores = [] 
    from tqdm import tqdm
    progress = tqdm(total=len(ds))
    with torch.no_grad():
        for i, obs in enumerate(ds):
            # if obs.y.sum() == len(obs.y): continue
            model.load_state_dict(ckpt['model'])
            model.eval()
            input_data = obs.x.cuda(non_blocking=True)
            edge_indices = obs.edge_index.cuda(non_blocking=True)
            edge_attr = obs.edge_attr.cuda(non_blocking=True)
            out = model(input_data, edge_indices, edge_attr)
            loss, true_positives, true_negatives, true_positives_analog, true_negatives_analog = loss_func(out, obs.y)

            basename = osp.basename(ds.raw_file_names[i].replace(".pt", ""))
            obsid = basename[1:11]
            if args["group"]:
                obsid = basename.split(".")[1][1:11]
            filename = osp.join(raw_dir, obsid, "pps", basename) if not args["test"] else osp.join(raw_dir, basename)
            scores.append({
                        "Filename": filename,
                        "TransNum": obs.y.sum().item(), 
                        "BkgNum": len(obs.y) - obs.y.sum().item(),
                        "Loss": loss.item(), 
                        "P_t": true_positives.item(), 
                        "N_t": true_negatives.item(), 
                        "aP_t": true_positives_analog.item(), 
                        "aN_t": true_negatives_analog.item()
                        })
            progress.update(1)


    scores = pd.DataFrame.from_records(scores)
    scores.to_csv("scores.csv", index=False)
    

    

    # print(list(generate_table(file) for file in json_files))
