import os.path as osp
from glob import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import logging
import pandas as pd
import torch
from transient_detection.DataPreprocessing.utilities import get_paired_filenames, get_uncompliant, in2d
from main_parser import parse

def check_compliance(args, genuine_pattern, simulated_pattern, rank=0, world_size=1):
    raw_dir           = args["PATHS"]["data"]
    keys              = args["Dataset"]["keys"]
    compliance_file   = args["PATHS"]["compliance_file"]

    gsnames = np.array(list(get_paired_filenames(raw_dir, genuine_pattern, simulated_pattern)))[rank::world_size]
    mode = 'w'
    if not args["check_compliance"]:
        mode = 'a'
        uncompliant_pairs = np.array(list(get_uncompliant(compliance_file)))
        gsnames = gsnames[np.logical_not(in2d(gsnames, uncompliant_pairs))]

    with open(compliance_file, mode) as f:
        for genuine, simulated in tqdm(gsnames, desc="processing {}/{}".format(rank, world_size), position=rank): 
            try:
                with fits.open(genuine, memmap=True) as gen_file, fits.open(simulated, memmap=True) as sim_file:
                    I_dat = Table(gen_file[1].data)
                    F_dat = Table(sim_file[1].data)
                tor_I_dat = torch.tensor(I_dat.to_pandas().values, dtype=torch.float32)
                tor_F_dat = torch.tensor(F_dat.to_pandas().values, dtype=torch.float32)
                if len(tor_I_dat) == 0 or len(tor_F_dat) == 0:
                    raise Exception() #this will be caught by the except statement
                if not torch.isfinite(tor_I_dat).all() or not torch.isfinite(tor_F_dat).all():
                    raise Exception() #this will be caught by the except statement

            except KeyboardInterrupt as ki:
                raise ki
            except:
                f.write(" ".join([genuine, simulated]) + "\n")
                f.flush()
                continue

            if any(key not in dat.colnames for dat in (I_dat, F_dat) for key in keys):
                f.write(" ".join([genuine, simulated]) + "\n")
                f.flush()
    
def test_compliance(args, genuine_pattern, simulated_pattern):
    raw_dir           = args["PATHS"]["data"]
    compliance_file   = args["PATHS"]["compliance_file"]

    # rfn_list = list(zip(sorted(glob(osp.join(raw_dir, genuine_pattern))), 
    #                glob(osp.join(raw_dir, simulated_pattern))
    #             ))
    rfn_list = np.array(list(get_paired_filenames(raw_dir, genuine_pattern, simulated_pattern)))
    with open(compliance_file, 'r') as f:
        uncompliant_list = f.read().split()
    uncompliant_genuine_list = [filename for filename in uncompliant_list if filename.endswith("EVLI0000.FTZ")]
    for genuine, simulated in tqdm(rfn_list): 
        if genuine in uncompliant_genuine_list:
            continue
        try:
            with fits.open(genuine, memmap=True) as gen_file, fits.open(simulated, memmap=True) as sim_file:
                I_dat = Table(gen_file[1].data)
                F_dat = Table(sim_file[1].data)
        except Exception as e:
            print(f"Got error while reading from files {(genuine, simulated)}.")
            logging.exception("Raised error is:")
            # raise e



if __name__ == "__main__":
    import multiprocessing
    args = parse()
    genuine_pattern   = "0*/pps/*EVLI0000.FTZ"
    simulated_pattern = "0*/pps/*EVLF0000.FTZ"
    pool = multiprocessing.Pool()
    ws = args["world_size"]
    for rank in range(ws):
        pool.apply_async(check_compliance, (args, genuine_pattern, simulated_pattern, rank, ws))
    pool.close()
    pool.join()
    pool = multiprocessing.Pool()
    for rank in range(ws):
        pool.apply_async(test_compliance, (args, genuine_pattern, simulated_pattern, rank, ws))
    pool.close()
    pool.join()