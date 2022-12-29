#!python
# """
# # Graph Neural Network for Simulated X-Ray Transient Detection
# The present work aims to train a GNN to label a particular sort of X-Ray transient using simulated events 
# overlayed onto real data from XMM-Newton observations. We will experiment with Graph Convolutional Networks (GCNs).
# We will therefore  have to trandsform our point-cloud data into a "k nearest neighbors"-type graph. 
# Data stored in the `raw` folder at the current working directory is taken from icaro.iusspavia.it 
# `/mnt/data/PPS_ICARO_SIM2`. Observations store data for each photon detected, with no filter applied, 
# in FITS files ending in `EVLI0000.FTZ` for the original observations and `EVLF0000.FTZ` for the observation 
# and simulation combined. We will refer to the former data as "genuine" and to the latter as "faked" for brevity.
# """
import os
import os.path as osp

import torch
from torch.utils.data import random_split
import torch_geometric.transforms as ttr
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from transient_detection.DataPreprocessing.data import FastIcaroDataset, IcaroDataset
from transient_detection.DeepLearning.models import GCNClassifier
from transient_detection.DeepLearning.utilities import train, test, log

from main_parser import parse

def main():
    args = parse()

    data_dict = args["data"]
    root_dir = data_dict["root_dir"]
    raw_dir = data_dict["raw_dir"]
    processed_dir = data_dict["processed_dir"]
    genuine_pattern = data_dict["genuine_pattern"]
    simulated_pattern = data_dict["simulated_pattern"]


    k_neighbors = data_dict["k_neighbors"]

    # root_dir="/home/scolombo/projects/rrg-lplevass/scolombo/data" #os.getcwd()
    if args["opts"]["fast"]:
        ds = FastIcaroDataset(root = processed_dir, 
                              pattern = simulated_pattern+".pt")
    else:
        ds = IcaroDataset(root = root_dir, 
                          genuine_pattern = genuine_pattern, 
                          simulated_pattern = simulated_pattern, 
                          raw_dir = raw_dir, 
                          processed_dir = processed_dir, 
                          pre_transform = ttr.KNNGraph(k=k_neighbors)
                         )

    model_dict = args["model"]

    batch_size = model_dict["batch_size"]
    num_hidden_channels = model_dict["num_hidden_channels"]
    num_layers = model_dict["num_layers"]
    split_fracs = model_dict["split_fracs"]
    lr = model_dict["learning_rate"]
    wd = model_dict["weight_decay"]
    epochs = model_dict["num_epochs"]


    model = GCNClassifier(num_layers = num_layers, 
                        input_dim  = ds.num_node_features, 
                        hidden_dim = num_hidden_channels, 
                        output_dim = ds.num_classes
                        )

    torch.cuda.set_device(args["opts"]["distributed_rank"])
    torch.distributed.init_process_group(
        backend="nccl", init_method=args["opts"]["distributed_init_method"], rank=args["opts"]["distributed_rank"])
    num_workers = args["opts"]["num_workers"]

    train_dataset, val_dataset, test_dataset = random_split(ds, split_fracs)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    device = torch.device(args["model"]["device_name"])
    model.to(device=device)
    model = DistributedDataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    os.makedirs("checkpoints", exist_ok=True)

    # Check if there are checkpoint files in the current directory
    checkpoint_files = [f for f in os.listdir("checkpoints") if f.startswith("checkpoint_")]
    if checkpoint_files:
        # Sort the checkpoint files by epoch
        checkpoint_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

        # Find the iteration value of the last epoch
        lastepoch = int(checkpoint_files[-1].split("_")[1].split(".")[0])

        # Load the last checkpoint
        model.load_state_dict(torch.load(checkpoint_files[-1]))
    else:
        lastepoch = 0
    
    for epoch in range(lastepoch + 1, lastepoch + epochs + 1):
        loss = train(model=model, train_loader=train_loader, optimizer=optimizer, device=device)
        train_acc, train_tp, train_fp = test(model=model, test_loader=train_loader, device=device)
        test_acc, test_tp, test_fp = test(model=model, test_loader=val_loader, device=device)
        log(logfile="logs.log",
            label=f"Epoch n. {epoch}:",
            forcemode='a',
            Epoch=epoch, 
            Loss=loss, 
            Train_accuracy=train_acc,
            Train_true_positives=train_tp,
            Train_false_positives=train_fp,
            Test_accuracy=test_acc,
            Test_true_positives=test_tp,
            Test_false_positives=test_fp
        )
        # Save the model state to a file
        torch.save(model.state_dict(), osp.join(os.getcwd(), "checkpoints", "checkpoint_{:05}.pt".format(epoch)))

    test_acc, test_tp, test_fp = test(model=model, test_loader=test_loader, device=device)

    log(logfile="logs.log",
        label=f"Final test",
        forcemode='a',
        Test_accuracy=test_acc,
        Test_true_positives=test_tp,
        Test_false_positives=test_fp
    )

if __name__ == "__main__":
    main()