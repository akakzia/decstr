import numpy as np
import torch
from torch.utils.data import Dataset
import itertools

def objects_distance(x, y):
    """
    A function that returns the euclidean distance between two objects x and y
    """
    assert x.shape == y.shape
    return np.linalg.norm(x - y)


def is_above(x, y):
    """
    A function that returns whether the object x is above y
    """
    assert x.shape == y.shape
    if np.linalg.norm(x[:2] - y[:2]) < 0.05 and 0.06 > x[2] - y[2] > 0.03:
        return 1.
    else:
        return -1.

def is_close(d):
    """ Return the value of the close predicate for a given distance between two pairs """
    if d < 0.09:
        return 1.
    else:
        return -1
class TripletDataset(Dataset):
    def __init__(self, anchor, positive, negative):

        assert anchor.shape[0] == positive.shape[0] == len(negative)  # negative is a list

        self.ids = np.arange(anchor.shape[0])
        np.random.shuffle(self.ids)

        self.anchor = anchor[self.ids].astype(np.float32)
        self.positive = positive[self.ids].astype(np.float32)
        self.negative = [negative[i].astype(np.float32) for i in self.ids]


    def __getitem__(self, index):
        idx = self.ids[index]
        return self.anchor[idx], self.positive[idx], self.negative[idx][0]


    def __len__(self):
        return self.ids.shape[0]

    def sample_data(self, bs):
        idxs = np.random.choice(self.ids, size=bs)
        return torch.Tensor(self.anchor[idxs]), torch.Tensor(self.positive[idxs]), torch.cat([torch.Tensor(self.negative[i][0]) for i in idxs])
        # return torch.Tensor(self.anchor[idxs]), torch.Tensor(self.positive[idxs]), [torch.Tensor(self.negative[i]) for i in idxs]

class TrajectoriesDataset(Dataset):
    def __init__(self, states, next_states, trajectories):

        assert states.shape[0] == next_states.shape[0] == len(trajectories)  # negative is a list

        self.ids = np.arange(states.shape[0])
        np.random.shuffle(self.ids)

        self.states = states[self.ids].astype(np.float32)
        self.next_states = next_states[self.ids].astype(np.float32)
        self.trajectories = [trajectories[i].astype(np.float32) for i in self.ids]


    def __getitem__(self, index):
        idx = self.ids[index]
        return self.states[idx], self.next_states[idx], self.trajectories[idx][0]


    def __len__(self):
        return self.ids.shape[0]

    def sample_data(self, bs):
        idxs = np.random.choice(self.ids, size=bs)
        return torch.Tensor(self.states[idxs]), torch.Tensor(self.next_states[idxs]), [torch.Tensor(self.trajectories[i]) for i in idxs]
        # return torch.Tensor(self.anchor[idxs]), torch.Tensor(self.positive[idxs]), [torch.Tensor(self.negative[i]) for i in idxs]


def get_configuration(positions):
    """
            This functions takes as input the positions of the objects in the scene and outputs the corresponding semantic configuration
            based on the environment predicates
            """
    positions = positions.reshape(5, -1)
    close_config = np.array([])
    above_config = np.array([])

    object_combinations = itertools.combinations(positions, 2)
    object_rel_distances = np.array([objects_distance(obj[0], obj[1]) for obj in object_combinations])

    close_config = np.array([is_close(distance) for distance in object_rel_distances])

    object_permutations = itertools.permutations(positions, 2)

    above_config = np.array([is_above(obj[0], obj[1]) for obj in object_permutations])

    res = np.concatenate([close_config, above_config])
    return res