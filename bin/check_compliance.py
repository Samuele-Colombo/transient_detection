import os.path as osp
from glob import glob
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import logging
from transient_detection.DataPreprocessing.utilities import get_paired_filenames, get_uncompliant, in2d
from main_parser import parse

def check_compliance(args, genuine_pattern, simulated_pattern):
    raw_dir           = args["PATHS"]["data"]
    keys              = args["Dataset"]["keys"]
    compliance_file   = args["PATHS"]["compliance_file"]

    gsnames = np.array(list(get_paired_filenames(raw_dir, genuine_pattern, simulated_pattern)))
    mode = 'w'
    if not args["check_compliance"]:
        mode = 'a'
        uncompliant_pairs = np.array(list(get_uncompliant(compliance_file)))
        gsnames = gsnames[np.logical_not(in2d(gsnames, uncompliant_pairs))]

    with open(compliance_file, mode) as f:
        for genuine, simulated in tqdm(gsnames): 
            try:
                with fits.open(genuine, memmap=True) as gen_file, fits.open(simulated, memmap=True) as sim_file:
                    I_dat = Table(gen_file[1].data)
                    F_dat = Table(sim_file[1].data)
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
    args = parse()
    genuine_pattern   = "0*/pps/*EVLI0000.FTZ"
    simulated_pattern = "0*/pps/*EVLF0000.FTZ"
    check_compliance(args, genuine_pattern, simulated_pattern)
    test_compliance(args, genuine_pattern, simulated_pattern)
