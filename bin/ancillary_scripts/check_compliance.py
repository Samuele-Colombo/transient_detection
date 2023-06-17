import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import logging
import pandas as pd
import torch
from transient_detection.DataPreprocessing.utilities import get_paired_filenames, get_uncompliant, in2d
from transient_detection.main_parser import parse

def check_compliance(gsnames, rank, world_size, keys):
    lines = []
    for genuine, simulated in tqdm(gsnames, desc="processing {}/{}".format(rank, world_size), position=rank): 
        try:
            with fits.open(genuine, memmap=True) as gen_file, fits.open(simulated, memmap=True) as sim_file:
                I_dat = Table(gen_file[1].data)
                F_dat = Table(sim_file[1].data)
            if len(F_dat) == 0 or len(I_dat) == 0:
                raise Exception() #this will be caught by the except statement
            for colname in keys:
                I_col = I_dat[colname]
                F_col = F_dat[colname]
                if np.issubdtype(I_col.dtype, np.floating):
                    assert np.isfinite(I_col).all() or not np.isfinite(F_col).all()

        except KeyboardInterrupt as ki:
            raise ki
        except:
            lines.append(" ".join([genuine, simulated]) + "\n")
            continue

    return lines
    
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
    ws = args["world_size"]
    raw_dir           = args["PATHS"]["data"]
    keys              = args["Dataset"]["keys"]
    compliance_file   = args["PATHS"]["compliance_file"]
    gsnames = np.array(list(get_paired_filenames(raw_dir, genuine_pattern, simulated_pattern)))

    mode = 'w'
    if not args["check_compliance"]:
        mode = 'a'
        uncompliant_pairs = np.array(list(get_uncompliant(compliance_file)))
        gsnames = gsnames[np.logical_not(in2d(gsnames, uncompliant_pairs))]
    
    # check_compliance(gsnames, 0, 1, mode)
    print("Checking {} files".format(len(gsnames)))
    pool = multiprocessing.Pool()
    lines_collection = pool.starmap(check_compliance, ((gsnames[rank::ws], rank, ws, keys) for rank in range(ws)))
    pool.close()
    pool.join()
    count = 0
    with open(compliance_file, mode) as f:
        for lines in lines_collection:
            f.writelines(lines)
            count += len(lines)

    print("wrote {} lines".format(count))
    
    # pool = multiprocessing.Pool()
    # for rank in range(ws):
    #     pool.apply_async(test_compliance, (args, genuine_pattern, simulated_pattern, rank, ws))
    # pool.close()
    # pool.join()