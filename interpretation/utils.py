import numpy as np
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, continuous, configs):

        assert continuous.shape[0] == configs.shape[0]

        self.ids = np.arange(continuous.shape[0])
        np.random.shuffle(self.ids)

        self.continuous = continuous[self.ids].astype(np.float32)
        self.configs = configs[self.ids].astype(np.float32)


    def __getitem__(self, index):
        idx = self.ids[index]
        return self.continuous[idx], self.configs[idx]


    def __len__(self):
        return self.ids.shape[0]