import torch
import pyg_lib #new in torch_geometric 2.2.0!
from torch_geometric.data import Data
from torch_geometric.data import Dataset

class IcaroData(Data):    
    @property
    def pos(self):
        return self.x[:, -3:]
    
    @pos.setter
    def pos(self, replace):
        assert replace.shape == self.pos.shape
        self.x[:, -3:] = replace


class IcaroDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, raw_dir=None, processed_dir=None):
        self._raw_dir       = raw_dir
        self._processed_dir = processed_dir
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_dir(self):
        if osp.isabs(self._raw_dir):
            return self._raw_dir
        return osp.join(os.getcwd(), self.raw_dir)

    @property
    def processed_dir(self):
        if osp.isabs(self._processed_dir):
            return self._processed_dir
        return osp.join(os.getcwd(), self._processed_dir)

    @property
    def raw_file_names(self):
        return list(sorted(list(glob(osp.join(self.raw_dir, "0*/pps/*EVLI0000.FTZ"))) +
                           list(glob(osp.join(self.raw_dir, "0*/pps/*EVLF0000.FTZ")))))

    @property
    def processed_file_names(self):
        return list(map(lambda name: osp.join(self.processed_dir, osp.basename(name)+".pt"), 
                        glob(osp.join(self.raw_dir, "0*/pps/*EVLF0000.FTZ"))))
    
    @property
    def num_classes(self):
        return 2

    def process(self):
        fnames = list(zip(sorted(glob(osp.join(self.raw_dir, "0*/pps/*EVLI0000.FTZ"))), 
                          sorted(glob(osp.join(self.raw_dir, "0*/pps/*EVLF0000.FTZ"))))
                     )
        for raw_path in fnames:
            # Read data from `raw_path`.
            dat = read_events(*raw_path)
            data = IcaroData(x  =torch.from_numpy(np.array([dat["PI"], dat["FLAG"], dat["TIME"], dat["X"], dat["Y"]]).T).float(), 
                             y  =torch.from_numpy(np.array(dat["ISFAKE"])).long())
            
            ss2 = StandardScaler()
            ss2.fit(data.pos)
            new_pos = ss2.transform(data.pos)
            data.pos = torch.tensor(new_pos)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, osp.basename(raw_path[-1])+".pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data