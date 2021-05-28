import numpy as np
from torch.utils.data import Dataset

class GraphDataset(Dataset):
    def __init__(self, continuous, p_close, p_above, shuffle=True):

        assert continuous.shape[0] == p_close.shape[0] == p_above.shape[0]

        self.ids = np.arange(continuous.shape[0])
        np.random.shuffle(self.ids)

        self.continuous = continuous[self.ids].astype(np.float32)
        self.p_close = p_close[self.ids].astype(np.float32)
        self.p_above = p_above[self.ids].astype(np.float32)


    def __getitem__(self, index):
        idx = self.ids[index]
        return self.continuous[idx], self.p_close[idx], self.p_above[idx]


    def __len__(self):
        return self.ids.shape[0]