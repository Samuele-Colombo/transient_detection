"""Mostly copy-pasted from https://github.com/ramyamounir/Template/blob/main/lib/utils/file.py"""

def checkdir(path, reset = True):

    import os, shutil

    if os.path.exists(path):
        if reset:
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)

def extract(file, path):
	
    import tarfile

    if file.endswith("tar.gz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path = path)
        tar.close()
    elif file.endswith("tar"):
        tar = tarfile.open(file, "r:")
        tar.extractall(path = path)
        tar.close()
    else:
        raise RuntimeError(f"Given file `{file}` is not a tar archive with standard extensions. Cannot extract.")

# copy-paste from https://github.com/facebookresearch/dino/blob/main/utils.py
def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise RuntimeError(f"Invalid value for a boolean flag: {s}")