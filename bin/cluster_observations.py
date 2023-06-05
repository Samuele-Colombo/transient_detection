import os.path as osp
from glob import glob
import pandas as pd

import numpy as np
import torch
from tqdm import tqdm

from transient_detection.DataPreprocessing.utilities import read_events
from main_parser import parse

def main():

    print("parsing arguments")
    args = parse()
    assert not args["test"], "This is not for test"
    torch.manual_seed(123)

    basedir = args["PATHS"]["data"]
    genuine_pattern   = osp.join("0*","pps","*EVLI0000.FTZ")
    simulated_pattern = osp.join("0*","pps","*EVLF0000.FTZ")
    genuine_pattern = osp.join(basedir, genuine_pattern)
    simulated_pattern = osp.join(basedir, simulated_pattern)
    # print(genuine_pattern, genuine_files)
    genuine_files = np.array(glob(genuine_pattern))[:5]
    simulated_files = np.array(glob(simulated_pattern))[:5]
    are_mos = np.ones_like(genuine_files, dtype=bool)
    for genuine_file, simulated_file, is_mos in tqdm(zip(genuine_files, simulated_files, are_mos), total=len(are_mos)):
        sim_directory, sim_orginal_file = osp.split(simulated_file)
        gen_directory, gen_orginal_file = osp.split(genuine_file)
        for group_num, cluster_data in cluster_data_generator(genuine_file, simulated_file, is_mos, args):
            cluster_data.write(osp.join(sim_directory, "group{:03}.{}".format(group_num, sim_orginal_file)), format='fits', overwrite=True)
            cluster_data[~cluster_data["ISSIMULATED"]].write(osp.join(gen_directory, "group{:03}.{}".format(group_num, gen_orginal_file)), format='fits', overwrite=True)


def cluster_data_generator(genuine_file, simulated_file, is_mos, args):
    lastcolname = "PHA" if is_mos else "TIME_RAW"
    keys =  args["Dataset"]["keys"]
    events = read_events(genuine_file, simulated_file, keys + [lastcolname])
    pd_events = events.to_pandas()[keys]
    # sim_events = events[events["ISSIMULATED"]].to_pandas().groupby(by=lastcolname, as_index=True, sort=True)
    sim_events = events[events["ISSIMULATED"]].to_pandas().set_index(lastcolname)
    for group_num in sim_events.index.unique():
        df = sim_events.loc[group_num, keys]
        if len(df.shape) == 1: continue #ignore single photon transients, since useless
        extremes = torch.tensor(np.vstack([df.max(axis=0).values, df.min(axis=0).values]))
        # extremes[:, 1:] = ss1.transform(extremes[:, 1:])
        # extremes[:, 0] = ss2.transform(extremes[:, 0])
        factors = torch.rand(2, len(keys)) + 1 
        extremes = extremes + (extremes - extremes[[1,0]]) * factors
        # mask = torch.all(torched_events >= extremes[1], dim=1) & torch.all(torched_events <= extremes[0], dim=1)
        mask = (pd_events >= extremes[1]) & (pd_events <= extremes[0])
        masked_events = events[mask.all(axis=1).values]
        yield group_num, masked_events
    print("ended")


if __name__ == "__main__":
    main()