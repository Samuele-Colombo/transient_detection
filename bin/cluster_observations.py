import sys
import os.path as osp
from glob import glob

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

from transient_detection.DataPreprocessing.utilities import read_events, get_paired_filenames, in2d, get_uncompliant
from main_parser import parse
import multiprocessing

def main():

    print("parsing arguments")
    args = parse()
    assert not args["test"], "This is not for test"
    torch.manual_seed(123)

    basedir = args["PATHS"]["data"]
    genuine_pattern   = osp.join("0*","pps","P*EVLI0000.FTZ")
    simulated_pattern = osp.join("0*","pps","P*EVLF0000.FTZ")
    genuine_pattern = osp.join(basedir, genuine_pattern)
    simulated_pattern = osp.join(basedir, simulated_pattern)
    
    # print(genuine_pattern, genuine_files)
    # genuine_files = np.array(glob(genuine_pattern))[:5]
    # simulated_files = np.array(glob(simulated_pattern))[:5]
    # are_mos = np.ones_like(genuine_files, dtype=bool)
    uncompliant_pairs = np.array(list(get_uncompliant(args["PATHS"]["compliance_file"])))
    gsnames = np.array(list(map(list, get_paired_filenames(basedir, genuine_pattern, simulated_pattern))))
    gsnames = gsnames[np.logical_not(in2d(gsnames, uncompliant_pairs))]

    # cluster_data(args, gsnames)

    pool = multiprocessing.Pool()
    ws = args["world_size"]
    for rank in range(ws):
        pool.apply_async(cluster_data, (args, gsnames, rank, ws))
    pool.close()
    pool.join()

def cluster_data(args, gsnames, rank=0, world_size=1):
    genuine_files, simulated_files = gsnames[rank::world_size].T
    are_mos = np.vectorize(lambda name: name.endswith("MIEVLI0000.FTZ"))(genuine_files)

    clustered_bar = tqdm(total=len(are_mos), position=rank, desc="Clustering {}/{}".format(rank, world_size))

    for genuine_file, simulated_file, is_mos in zip(genuine_files, simulated_files, are_mos):
        sim_directory, sim_orginal_file = osp.split(simulated_file)
        gen_directory, gen_orginal_file = osp.split(genuine_file)
        for group_num, cluster_data in cluster_data_generator(genuine_file, simulated_file, is_mos, args):
            new_sim_file = osp.join(sim_directory, "group{:03}.{}".format(group_num, sim_orginal_file))
            new_gen_file = osp.join(gen_directory, "group{:03}.{}".format(group_num, gen_orginal_file))
            if not osp.exists(new_gen_file) or not args["fast"]:
                cluster_data[~cluster_data["ISSIMULATED"]].write(new_gen_file, format='fits', overwrite=True)
            if not osp.exists(new_sim_file) or not args["fast"]:
                cluster_data.write(new_sim_file, format='fits', overwrite=True)
        clustered_bar.update(1)
        sys.stdout.flush()
    clustered_bar.close()


def cluster_data_generator(genuine_file, simulated_file, is_mos, args):
    lastcolname = "PHA" if is_mos else "TIME_RAW"
    keys =  args["Dataset"]["keys"]
    events = read_events(genuine_file, simulated_file, keys + [lastcolname])
    pd_events = events.to_pandas()[keys]
    # sim_events = events[events["ISSIMULATED"]].to_pandas().groupby(by=lastcolname, as_index=True, sort=True)
    sim_events = events[events["ISSIMULATED"]].to_pandas().set_index(lastcolname)
    for group_num in sim_events.index.unique():
        group_num = int(group_num)
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
        new_keys = ["TIME", "X", "Y"]
        for key in new_keys:
            me_min = masked_events[key].min()
            me_max = masked_events[key].max()
            masked_events[key] = (masked_events[key] - me_min)/(me_max-me_min)
            assert np.all(np.logical_and(masked_events[key] >= 0, masked_events[key] <= 1)), "masked events not finite {}".format(masked_events)

        # pd_masked_events = masked_events.to_pandas()[new_keys]
        # new_extremes = torch.tensor(np.vstack([pd_masked_events.max(axis=0).values, pd_masked_events.min(axis=0).values]))
        # masked_events[new_keys].values = pd_masked_events = (torch.tensor(pd_masked_events.values) - new_extremes[1])/(new_extremes[0] - new_extremes[1])
        # assert np.logical_and(pd_masked_events >= 0, pd_masked_events <= 1).all(), str(pd_masked_events)
        # for colname in new_keys:
        #     assert np.all(np.logical_and(masked_events[colname] >= 0, masked_events[colname] <= 1)), "masked events not finite {}".format(masked_events)
        # print(genuine_file, ": group", group_num)
        yield int(group_num), masked_events


if __name__ == "__main__":
    main()