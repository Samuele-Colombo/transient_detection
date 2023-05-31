import os
import os.path as osp
from glob import glob
import functools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import datetime

import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp

from main_parser import parse
from transient_detection.DeepLearning.models import GCNClassifier
from transient_detection.DataPreprocessing.data import FastSimTransientDataset

def scatter_from_data(data, ax, args):
    
    # Extract the individual columns
    is_event = data.y.squeeze()
    keys = dict(zip(args["Dataset"]["keys"], range(len(args["Dataset"]["keys"]))))
    
    # Separate the data points based on the label (background or event)
    background_points = data.x[is_event == 0]
    event_points = data.x[is_event == 1]
    
    # Set the size of the points based on PI
    background_sizes = torch.clip(background_points[:, keys["PI"]], 1, 100)
    event_sizes = torch.clip(event_points[:, keys['PI']], 1, 100)

    ax.set_title("Fixed Scatterplot")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    
    # Plot the background points
    ax.scatter(background_points[:, keys['X']], background_points[:, keys['TIME']], background_points[:, keys['Y']],
               c='b', label='Background', alpha=0.6, s=background_sizes)
    
    # Plot the event points
    ax.scatter(event_points[:, keys['X']], event_points[:, keys['TIME']], event_points[:, keys['Y']],
               c='r', label='Event', alpha=0.6, s=event_sizes)
    
    if ax.get_legend() is None:
        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Time')
        ax.set_zlabel('Y')
        
        # Add a legend
        ax.legend()

    return ax

def line_from_logs(json_file, ax):
    df = pd.read_json(json_file, lines=True)

    # Sort the data based on the 'Epoch' value
    df.sort_values(by='Epoch', inplace=True)

    # Separate data for validation=True and validation=False
    validation_true = df[df['Validation'] == 'True']
    validation_false = df[df['Validation'] == 'False']

    # Extract x, y, and z values for each dataset
    x_true_analog = validation_true['true_positives_analog'].str.split(' ', expand=True)[0].astype(float)
    y_true_analog = validation_true['true_negatives_analog'].str.split(' ', expand=True)[0].astype(float)
    z_true = validation_true['loss'].str.split(' ', expand=True)[0].astype(float)

    x_false_analog = validation_false['true_positives_analog'].str.split(' ', expand=True)[0].astype(float)
    y_false_analog = validation_false['true_negatives_analog'].str.split(' ', expand=True)[0].astype(float)
    z_false = validation_false['loss'].str.split(' ', expand=True)[0].astype(float)

    x_true = validation_true['true_positives'].str.split(' ', expand=True)[0].astype(float)
    y_true = validation_true['true_negatives'].str.split(' ', expand=True)[0].astype(float)

    x_false = validation_false['true_positives'].str.split(' ', expand=True)[0].astype(float)
    y_false = validation_false['true_negatives'].str.split(' ', expand=True)[0].astype(float)

    lossfunc = lambda X, Y: 1 - X * Y

    scatter_true_analog = ax.plot(x_true_analog, y_true_analog, z_true, color='blue', alpha=0.7, label="Validation Loss (Analog)", linestyle="--")[0]

    scatter_true = ax.plot(x_true, y_true, lossfunc(x_true, y_true), color='blue', alpha=0.7, label="Validation Loss")[0]

    # Create a scatter plot for validation=False
    scatter_false_analog = ax.plot(x_false_analog, y_false_analog, z_false, color='red', alpha=0.7, label="Training Loss (Analog)", linestyle="--")[0]

    scatter_false = ax.plot(x_false, y_false, lossfunc(x_false, y_false), color='red', alpha=0.7, label="Training Loss")[0]

    # Create a meshgrid for the function
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = lossfunc(X, Y)

    # Create a surface plot for the function
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)

    ax.legend()
    
    lines = (scatter_true_analog, scatter_true, scatter_false_analog, scatter_false)
    arrays = (x_true_analog, y_true_analog, x_true, y_true, z_true, x_false_analog, y_false_analog, x_false, y_false, z_false)
    return ax, lines, arrays


def main():
    args = parse()

    # Set up the figure and subplots
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')

    # Subplot 1: Fixed 3D scatterplot of uniform random points
    current_device = 0
    processed_dir = args["PATHS"]["processed_data"]
    simulated_pattern = osp.join("0*","pps","*EVLF0000.FTZ")
    if args["test"]:
        simulated_pattern = "*.evt.fits"

    data = FastSimTransientDataset(root = processed_dir, 
                                 pattern = osp.basename(simulated_pattern)+".pt",
                                 device="cuda:{}".format(current_device)
                                )[0].cpu()

    ax1 = scatter_from_data(data, ax1, args)

    # Subplot 2: Animated sequence of uniformly distributed points
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title("Animated Scatterplot")

    # Initialize scatter object for the animation
    scatter_anim = ax2.scatter([], [], [], c='b')

    # Text for displaying the frame number
    frame_text = ax2.text(0.8, 0.8, 0.9, '', transform=ax2.transAxes)

    num_hidden_channels = args["Model"]["hidden_dim"]
    num_layers = args["Model"]["num_layers"]


    print('Initializing Process Group...')
    #init the process group
    mp.set_start_method("spawn")
    timeout = datetime.timedelta(hours=1)
    dist.init_process_group(backend=args["dist_backend"], init_method=args["distributed_init_method"], world_size=1, rank=0, timeout=timeout)

    model = GCNClassifier(num_layers = num_layers, 
                        input_dim  = data.num_node_features, 
                        hidden_dim = num_hidden_channels, 
                        output_dim = 1
                        )
    model = DistributedDataParallel(model.cuda())#, device_ids=[current_device], output_device=current_device)
    
    ckpts = sorted(glob(osp.join(args["PATHS"]["out"], "weights", args["Model"]["model"], "Epoch_*.pth")))

    # Function to update the animated scatterplot
    def update(frame):
        ckpt = torch.load(ckpts[frame], map_location='cpu')
        epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        model.eval()
        input_data = data.x.cuda(non_blocking=True)
        edge_indices = data.edge_index.cuda(non_blocking=True)
        edge_attr = data.edge_attr.cuda(non_blocking=True)
        out = model(input_data, edge_indices, edge_attr)
        out = torch.nn.functional.sigmoid(out)
        pred = out.round().bool()
        data.y = pred

        # Generate new set of random points for each frame
        ax = scatter_from_data(data.cpu(), ax2, args)
        frame_text.set_text(f"Epoch: {epoch}")

    json_file = args["PATHS"]["loss_logfile"]

    # Subplot 3: 3D multi-line plot with gradually revealed points
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    # Set labels for the axes
    ax3.set_xlabel('True Positives')
    ax3.set_ylabel('True Negatives')
    ax3.set_zlabel('Loss')

    # Set the plot title
    ax3.set_title('Loss evolution of GCNN training and validation')

    # Create the legend

    # ax3.set_xlim(0, 1)
    # ax3.set_ylim(0, 1)
    # ax3.set_zlim(0, 1)

    ax3, lines, arrays = line_from_logs(json_file, ax3)

    def update_lines(frame, lines, arrays):
        scatter_true_analog, scatter_true, scatter_false_analog,scatter_false = lines 
        x_true_analog, y_true_analog, x_true, y_true, z_true, x_false_analog, y_false_analog, x_false, y_false, z_false = arrays 
        lossfunc = lambda X, Y: 1 - X * Y

        # Update the scatter plot positions and colors
        scatter_true_analog._offsets3d = (x_true_analog[:frame+1], y_true_analog[:frame+1], z_true[:frame+1])
        scatter_true._offsets3d = (x_true[:frame+1], y_true[:frame+1], lossfunc(x_true[:frame+1], y_true[:frame+1]))
        scatter_false_analog._offsets3d = (x_false_analog[:frame+1], y_false_analog[:frame+1], z_false[:frame+1])
        scatter_false._offsets3d = (x_false[:frame+1], y_false[:frame+1], lossfunc(x_false[:frame+1], y_false[:frame+1]))

    ax3.view_init(azim=45, elev=30)

    # Total number of frames and duration (adjust as needed)
    total_frames = len(ckpts)
    duration = 4000  # in milliseconds

    # Create the animations for each subplot
    anim2 = animation.FuncAnimation(fig, update, frames=total_frames, interval=duration/total_frames)
    anim3 = animation.FuncAnimation(fig, functools.partial(update_lines, lines=lines, arrays=arrays), frames=total_frames, interval=duration/total_frames)

    # Save the animations as GIFs
    # anim2.save('animated_scatterplot.gif', writer='pillow')
    # anim3.save('gradual_multi_line_plot.gif', writer='pillow')

    # Show the subplots (optional)
    plt.show()

if __name__=="__main__":
    main()