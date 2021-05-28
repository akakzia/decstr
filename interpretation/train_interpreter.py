import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from utils import GraphDataset
from collections import defaultdict
from models import SimpleCVAE

def load_data(n=3, minimal=False):
    """
    Load the dataset of configurations and geometric states
    Return the loaded data in the form of a graph
    Each node contains geometric states of a specific object
    Each predicate is associated with an edge
    n: number of objects
    minimal: if true, returns only predicates between two objects (only two nodes are considered)
    """
    n_comb = n * (n - 1) // 2
    path_config_transitions_reached = '/home/ahmed/Documents/DECSTR/ICLR2021_version/decstr/language/data/learned_configs_continuous.pkl'
    with open(path_config_transitions_reached, 'rb') as f:
        config_states = pickle.load(f)
    configs = np.concatenate([config_states[:, 0, :], config_states[:, 1, :]])
    states = np.concatenate([config_states[:, 2, :], config_states[:, 3, :]])

    states = states.reshape(states.shape[0], n, 3)
    p_close = configs[:, :n_comb].reshape(configs.shape[0], n, 1).repeat(2, axis=-1)
    p_above = np.stack([configs[:, [3, 5, 7]], configs[:, [4, 6, 8]]], axis=-1)
    if minimal:
        return states[:, :2], p_close[:, 0], p_above[:, 0]
    return states, p_close, p_above

def split_data(df, train_prop=0.6, test_prop=0.2):
    """
    Splits data into three portions
    """
    # Make sure something is left for held out
    assert train_prop + test_prop < 1.

    data_len = df[0].shape[0]
    ids = np.arange(data_len)
    np.random.shuffle(ids)
    train_marker = int(data_len * train_prop)
    test_marker = int(data_len * (train_prop + test_prop))
    train_data = [d[ids[:train_marker]] for d in df]
    test_data = [d[ids[train_marker:test_marker]] for d in df]
    validation_data = [d[ids[test_marker:]] for d in df]
    return train_data, test_data, validation_data

if __name__ == '__main__':
    batch_size = 128
    k_param = 0.6
    learning_rate = 0.005
    epochs = 150

    for k in range(5):
        print('Seed {} / 5'.format(k + 1))
        seed = np.random.randint(1e6)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data = load_data(minimal=True)
        data_train, data_test, data_validation = split_data(data, train_prop=0.05, test_prop=0.8)
        dataset = GraphDataset(data_train[0], data_train[1], data_train[2], shuffle=True)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        vae = SimpleCVAE()

        logs = defaultdict(list)

        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)


        def loss_fn(recon_x, x, mean, log_var):
            bce = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return (bce + k_param * kld) / x.size(0)


        for epoch in range(epochs + 1):
            for iteration, (states, p_close, p_above) in enumerate(data_loader):
                # TODO device add
                recon_state, mean, log_var, z = vae(states, p_close)

                target = p_close[:, 0].unsqueeze(-1)
                loss = loss_fn(recon_state, target, mean, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logs['loss'].append(loss.item())
                stop = 1

        s = torch.Tensor(data_test[0])
        p1 = data_test[1][:, 0].astype(np.int)
        x = (vae.inference(s, n=1).detach().numpy() > 0.5).astype(np.int)[:, 0]
        print('Precision: {}'.format(np.sum(x == p1) / x.shape[0]))
