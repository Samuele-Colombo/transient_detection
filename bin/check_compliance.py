import os.path as osp
from glob import glob
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
from main_parser import parse

def check_compliance(args):
    raw_dir           = args["PATHS"]["data"]
    genuine_pattern   = args["PATHS"]["genuine_pattern"]
    simulated_pattern = args["PATHS"]["simulated_pattern"]
    keys              = args["Dataset"]["keys"]

    rfn_list = list(zip(glob(osp.join(raw_dir, genuine_pattern)), 
                   glob(osp.join(raw_dir, simulated_pattern))
                ))
    with open(args["PATHS"]["compliance_file"], 'w') as f:
        for genuine, simulated in tqdm(rfn_list): 
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

            # I_dat = astropy_table.Table.read(genuine, hdu=1, format=)
            # F_dat = astropy_table.Table.read(simulated, hdu=1)
            if any(key not in dat.colnames for dat in (I_dat, F_dat) for key in keys):
                f.write(" ".join([genuine, simulated]) + "\n")
                f.flush()
    
def test_compliance(args):
    raw_dir           = args["PATHS"]["data"]
    genuine_pattern   = args["PATHS"]["genuine_pattern"]
    simulated_pattern = args["PATHS"]["simulated_pattern"]
    keys              = args["Dataset"]["keys"]

    rfn_list = list(zip(glob(osp.join(raw_dir, genuine_pattern)), 
                   glob(osp.join(raw_dir, simulated_pattern))
                ))
    with open(args["PATHS"]["compliance_file"], 'w') as f:
        for genuine, simulated in tqdm(rfn_list): 
            with fits.open(genuine, memmap=True) as gen_file, fits.open(simulated, memmap=True) as sim_file:
                I_dat = Table(gen_file[1].data)
                F_dat = Table(sim_file[1].data)


if __name__ == "__main__":
    args = parse()
    check_compliance(args)
    test_compliance(args)
