import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
from transient_detection.DataPreprocessing.utilities import get_paired_filenames, get_uncompliant, in2d
from transient_detection.main_parser import parse

def get_sig_to_noise(gsnames, rank):
    # gsnames = gsnames[rank::ws]
    sig_to_noise = np.empty(len(gsnames))
    count = 0
    for genuine, simulated in tqdm(gsnames, position=rank): 
        with fits.open(genuine, memmap=True) as gen_file, fits.open(simulated, memmap=True) as sim_file:
            I_dat = Table(gen_file[1].data)
            F_dat = Table(sim_file[1].data)
        sig_to_noise[count] = float(len(F_dat))/len(I_dat) - 1.
        count += 1
    return sig_to_noise
    
if __name__ == "__main__":
    import multiprocessing
    import matplotlib.pyplot as plt
    from itertools import repeat
    args = parse()
    genuine_pattern   = "0*/pps/group*P*EVLI0000.FTZ"
    simulated_pattern = "0*/pps/group*P*EVLF0000.FTZ"
    raw_dir           = args["PATHS"]["data"]
    keys              = args["Dataset"]["keys"]
    compliance_file   = args["PATHS"]["compliance_file"]

    pool = multiprocessing.Pool()
    ws = args["world_size"]

    gsnames = np.array(list(get_paired_filenames(raw_dir, genuine_pattern, simulated_pattern)))
    uncompliant_pairs = np.array(list(get_uncompliant(compliance_file)))
    gsnames = gsnames[np.logical_not(in2d(gsnames, uncompliant_pairs))]
    pool = multiprocessing.Pool()
    ws = args["world_size"]
    print("analysing {} files".format(len(gsnames)))
    sig_to_noise_2d = pool.starmap(get_sig_to_noise, ((gsnames[rank::ws], rank) for rank in range(ws)))
    # sig_to_noise_2d = [get_sig_to_noise(gsnames, 0, 1)]
    pool.close()
    pool.join()
    sig_to_noise_vec = []
    for result in sig_to_noise_2d:
        sig_to_noise_vec.extend(result)

    bins = 25

    # Generate the histogram using numpy
    # hist, bins = np.histogram(sig_to_noise_vec, bins=bins)

    # Plot the histogram using matplotlib
    plt.hist(sig_to_noise_vec, bins=bins, log=True)
    plt.xlabel('Signal to Noise Ratio')
    plt.ylabel('Count')
    plt.title('Distribution of Signal to Noise Ratios')
    plt.savefig("S2NR.plot.png")