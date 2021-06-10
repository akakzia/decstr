import numpy as np
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    def __init__(self, anchor, positive, negative):

        assert anchor.shape[0] == positive.shape[0] == negative.shape[0]

        self.ids = np.arange(anchor.shape[0])
        np.random.shuffle(self.ids)

        self.anchor = anchor[self.ids].astype(np.float32)
        self.positive = positive[self.ids].astype(np.float32)
        self.negative = negative[self.ids].astype(np.float32)


    def __getitem__(self, index):
        idx = self.ids[index]
        return self.anchor[idx], self.positive[idx], self.negative[idx]


    def __len__(self):
        return self.ids.shape[0]