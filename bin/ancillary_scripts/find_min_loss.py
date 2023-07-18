import json
import numpy as np
import pandas as pd
import os.path as osp

def generate_table(filename):
    filedir =  osp.split(osp.split(filename)[0])[1].replace("out_",'')
    with open(filename, 'r') as file:
        data=pd.DataFrame(json.loads(line) for line in file)
    data["Run"] = np.full_like(data.iloc[:, 1], filedir)
    data["Epoch"] = data["Epoch"].astype(int) + 1
    data["Validation"] = data["Validation"].astype(bool)
    for column in data.columns:
        if column in ["Run", "Epoch", "Validation"]: continue
        data[column] = data[column].apply(lambda x: x.split()[0]).astype(float)
    data.set_index("Run")
    return data

if __name__ == "__main__":
    from glob import glob
    import os.path as osp
    json_files = np.array(list(glob(osp.join("test", "Icaro", "out_*","*json"))))
    data = pd.concat(list(generate_table(file) for file in json_files), axis=0)
    data = data.loc[data["Validation"]].drop("Validation", axis=1)
    grouped = data.groupby("Run").apply(lambda x: x.loc[x['loss'].idxmin()])
    grouped["Run"] = grouped["Run"].apply(lambda x: x.replace("small", "").replace(".adagrad", "").replace( ".modloss", "").replace( ".simpler", "").replace("-", "").replace(".", ""))
    grouped.set_index("Run", inplace=True)
    print(grouped.style.to_latex())
    # print(list(generate_table(file) for file in json_files))
