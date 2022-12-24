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
import torch_geometric.transforms as ttr
from torch.utils.data import random_split

from transient_detection.DataPreprocessing.data import IcaroDataset
from transient_detection.DeepLearning.models import GCNClassifier

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
    ds = IcaroDataset(root = root_dir, 
                      genuine_pattern=genuine_pattern, 
                      simulated_pattern=simulated_pattern, 
                      raw_dir=raw_dir, 
                      processed_dir=processed_dir, 
                      pre_transform = ttr.KNNGraph(k=k_neighbors)
                     )

    model_dict = args["model"]

    learning_rate = model_dict["learning_rate"]
    device_name = model_dict["device_name"]
    batch_size = model_dict["batch_size"]
    num_epochs = model_dict["num_epochs"]
    num_hidden_channels = model_dict["num_hidden_channels"]
    num_layers = model_dict["num_layers"]
    split_fracs = model_dict["split_fracs"]

    model = GCNClassifier(num_layers = num_layers, 
                        input_dim  = ds.num_node_features, 
                        hidden_dim = num_hidden_channels, 
                        output_dim = ds.num_classes
                        )

    train_dataset, val_dataset, test_dataset = random_split(ds, split_fracs)
