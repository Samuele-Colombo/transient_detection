import os, os.path as osp
from glob import glob
basedir = "C:\\Users\\samuc\\Work\\transient_detection\\test\\icaro\\raw"
pattern = "0*\\pps\\P*.FTZ"                     
timing_pattern = "0*\\pps\\P*S00*TIEVL*0000.FTZ"
strict_pattern = "0*\\pps\\P*S00*IEVL*0000.FTZ"
timing_filenames = glob(osp.join(basedir, timing_pattern))
for filename in timing_filenames:
    os.remove(filename)

filenames = glob(osp.join(basedir, pattern))
strict_filenames = glob(osp.join(basedir, strict_pattern)) 
for filename in filenames:
    if filename not in strict_filenames:
            try:
                os.remove(filename)
            except:
                print(filename)
dirs = glob(osp.join(basedir,"0*"))
for dir in dirs:                     
    dirpps = osp.join(dir,"pps")
    l = len(os.listdir(dirpps))
    if l == 0:
        os.rmdir(dirpps)
        os.rmdir(dir)
    elif l < 6:
        for file in os.listdir(dirpps):
            os.remove(osp.join(dirpps,file))
        os.rmdir(dirpps)
        os.rmdir(dir)

filenames = glob(osp.join(basedir, pattern))
dirs = glob(osp.join(basedir,"0*"))
print("files - dirs * 6: ", len(filenames)-len(dirs)*6)
if len(filenames)-len(dirs)*6 > 0:
    for dir in dirs:                     
        dirpps = osp.join(dir,"pps")
        l = len(os.listdir(dirpps))
        if l > 6:
            print(dir)
            for file in os.listdir(dirpps):
                print("\t",file)