import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pickle
from utils import TripletDataset
from collections import defaultdict
from models import SimpleCVAE, Hulk, to_categorical
import env
import time
import gym


def load_data():
    """
    Load the dataset of configurations and geometric states
    """
    path_states_configs = '/home/ahmed/Documents/DECSTR/ICLR2021_version/decstr/interpretation/triplet_data.pkl'
    with open(path_states_configs, 'rb') as f:
        anchor, positive, negative = pickle.load(f)

    return anchor, positive, negative

def split_data(df, train_prop=0.6):
    """
    Splits data into three portions
    """
    # Make sure something is left for test
    assert train_prop < 1.

    data_len = df[0].shape[0]
    ids = np.arange(data_len)
    np.random.shuffle(ids)
    train_marker = int(data_len * train_prop)
    train_data = [d[ids[:train_marker]] for d in df]
    test_data = [d[ids[train_marker:]] for d in df]
    return train_data, test_data


def split_pos_neg(df):
    """
    Takes as input a dataset of configurations
    Constructs three sets of anchor, positive and negative according to configus
    """
    anchor = []
    positive = []
    negative = []
    config_to_ids = {}
    ids_to_config = {}
    for j, c in enumerate(df[-1]):
        try:
            config_to_ids[str(c)].append(j)
        except KeyError:
            config_to_ids[str(c)] = [j]
        ids_to_config[j] = str(c)

    for j, c in enumerate(df[-1]):
        anchor.append(df[0][j])
        positive.append(df[0][np.random.choice(config_to_ids[str(c)])])
        neg_c = np.random.choice(list(config_to_ids.keys()))
        while neg_c == str(c):
            neg_c = np.random.choice(list(config_to_ids.keys()))
        negative.append(df[0][np.random.choice(config_to_ids[str(neg_c)])])
    with open('triplet_data.pkl', 'wb') as f:
        pickle.dump((np.array(anchor), np.array(positive), np.array(negative)), f)
    return np.array(anchor), np.array(positive), np.array(negative)

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

        data_anchor, data_positive, data_negative = load_data()
        # data_train, data_test = split_data(data, train_prop=0.1)
        # train_anchor, train_positive, train_negative = split_pos_neg(data)
        dataset = TripletDataset(data_anchor, data_positive, data_negative)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        # vae = SimpleCVAE(state_size=9, latent_size=27)
        vae = Hulk(state_size=3, latent_size=2)

        logs = defaultdict(list)

        optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)
        optimizer_info = torch.optim.Adam(vae.parameters(), lr=learning_rate)


        def loss_fn(recon_x, x_pos, x_neg, mean, log_var):
            mse_pos = torch.nn.functional.mse_loss(recon_x, x_pos, reduction='mean')
            mse_neg = torch.nn.functional.mse_loss(recon_x, x_neg, reduction='mean')
            kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return (mse_pos - 0.7 * mse_neg + k_param * kld) / recon_x.size(0)


        for epoch in range(epochs + 1):
            for iteration, (states, positives, negatives) in enumerate(data_loader):
                # recon_state_p1, mean_p1, log_var_p1, z_p1, \
                # recon_state_p2, mean_p2, log_var_p2, z_p2 = vae(states, p_close, p_above)
                recon_state, mean, log_var, z = vae(states)

                target_positives = positives.reshape(positives.size(0), positives.size(1) * positives.size(2))
                target_negatives = negatives.reshape(negatives.size(0), negatives.size(1) * negatives.size(2))
                loss = loss_fn(recon_state, target_positives, target_negatives, mean, log_var)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logs['loss'].append(loss.item())
                stop = 1
            print('Epoch {} / {} ---- | Loss = {}'.format(epoch, epochs, loss.item()))

        # x1 = vae.inference(n=120).detach().numpy().reshape(40, 3, 6)
        x1 = vae.inference(n=120).detach().numpy().reshape(120, 3, 3)
        env = gym.make('FetchManipulate3Objects-v0')
        for i in range(10):
            env.reset_positions(x1[i])
            env.render()
            time.sleep(1)
            stop = 1
