import os.path as osp
from glob import glob
import pandas as pd

import numpy as np

from transient_detection.DataPreprocessing.utilities import read_events
from main_parser import parse

def main():

    print("parsing arguments")
    args = parse()
    basedir = args["PATHS"]["data"]
    genuine_pattern   = osp.join("0*","pps","*EVLI0000.FTZ")
    simulated_pattern = osp.join("0*","pps","*EVLF0000.FTZ")
    if args["test"]:
        genuine_pattern   = "*.bkg.fits"
        simulated_pattern = "*.evt.fits"
    genuine_pattern = osp.join(basedir, genuine_pattern)
    simulated_pattern = osp.join(basedir, simulated_pattern)
    # print(genuine_pattern, genuine_files)
    if not args["test"]:
        genuine_files = np.array(glob(genuine_pattern))
        simulated_files = np.array(glob(simulated_pattern))
        are_mos = np.vectorize(lambda name: name.endswith("MIEVLI0000.FTZ"))(genuine_files)
    else:
        genuine_files = np.array(glob(genuine_pattern))[:5]
        simulated_files = np.array(glob(simulated_pattern))[:5]
        are_mos = np.ones_like(genuine_files, dtype=bool)
    simobs = []
    for gfile, sfile, is_mos in zip(genuine_files, simulated_files, are_mos):
        if args["test"]:
            lastcolname="TIME"
            events = read_events(gfile, sfile, args["Dataset"]["keys"]+["ISEVENT"])
            sim_events = events[events["ISSIMULATED"]].to_pandas().drop("ISSIMULATED", axis=1).set_index(lastcolname).sort_index()
        else:
            lastcolname = "PHA" if is_mos else "TIME_RAW"
            events = read_events(gfile, sfile, args["Dataset"]["keys"] + [lastcolname])
            # sim_events = events[events["ISSIMULATED"]].to_pandas().groupby(by=lastcolname, as_index=True, sort=True)
            sim_events = events[events["ISSIMULATED"]].to_pandas().set_index(lastcolname).sort_index()
        simobs.append(sim_events)
    for simob in simobs:
        print(simob)
        # print(simob["ISEVENT"].all())
        print()

if __name__ == "__main__":
    main()