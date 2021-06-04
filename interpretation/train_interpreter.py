import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pickle
from utils import GraphDataset
from collections import defaultdict
from models import SimpleCVAE, Hulk, to_categorical
import env
import time
import gym


def load_data(n=3, minimal=False):
    """
    Load the dataset of configurations and geometric states
    """
    path_states_configs = '/home/ahmed/Documents/DECSTR/ICLR2021_version/decstr/interpretation/states_configs_no_rot.pkl'
    with open(path_states_configs, 'rb') as f:
        state, configs = pickle.load(f)

    return state, configs

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
    batch_size = 256
    k_param = 0.6
    learning_rate = 0.005
    epochs = 60

    for k in range(1):
        print('Seed {} / 5'.format(k + 1))
        seed = np.random.randint(1e6)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data = load_data()
        data_train, data_test, data_validation = split_data(data, train_prop=0.1, test_prop=0.2)
        dataset = GraphDataset(data_train[0], data_train[1])
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        # vae = SimpleCVAE(state_size=9, latent_size=27)
        vae = Hulk(state_size=9, latent_size=27)

        logs = defaultdict(list)

        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
        optimizer_info = torch.optim.Adam(vae.parameters(), lr=learning_rate)


        def loss_fn(recon_x, x, mean, log_var):
            mse = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
            kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return (mse + k_param * kld) / x.size(0)


        for epoch in range(epochs + 1):
            for iteration, (states, configs) in enumerate(data_loader):
                # TODO device add
                # recon_state_p1, mean_p1, log_var_p1, z_p1, \
                # recon_state_p2, mean_p2, log_var_p2, z_p2 = vae(states, p_close, p_above)
                recon_state, mean, log_var, z = vae(states)

                target = states.reshape(states.size(0), states.size(1) * states.size(2))
                # target = states[:, 0]
                # target = torch.cat([states[:, 0], states[:, 0], abs(states[:, 0] - states[:, 1])], dim=-1)
                loss = loss_fn(recon_state, target, mean, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logs['loss'].append(loss.item())
                stop = 1
            print('Epoch {} / {} ---- | Loss = {}'.format(epoch, epochs, loss.item()))

        s = torch.Tensor(data_test[0])

        # x1 = vae.inference(n=120).detach().numpy().reshape(40, 3, 6)
        x1 = vae.inference(n=120).detach().numpy().reshape(120, 3, 3)
        env = gym.make('FetchManipulate3Objects-v0')
        for i in range(10):
            env.reset_positions(x1[i])
            env.render()
            time.sleep(1)
            stop = 1
