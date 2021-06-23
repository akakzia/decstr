import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import pickle
from utils import TripletDataset
from collections import defaultdict
from models import SimpleModel, GnnModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


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

def plot_scatter_test(states, config_to_ids, configs_to_color):
    model.eval()
    pca = PCA(n_components=2)
    out = model.network(model.W(torch.Tensor(states))).detach().numpy()
    pca.fit(out)
    fig = plt.figure()
    ax = fig.add_subplot()
    cmap = get_cmap(len(list(config_to_ids.keys())))
    for i, c in enumerate(config_to_ids.keys()):
        input_state = torch.Tensor(states[config_to_ids[str(c)]])
        embeddings = (model.network(model.W(input_state)).detach().numpy())
        y = pca.transform(embeddings)
        ax.scatter(y[:, 0], y[:, 1], color=cmap(i), label=str(c))
    plt.legend(title='legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.savefig('images/{}.jpg'.format(epoch), bbox_inches='tight')
    plt.close('all')
    model.train()


if __name__ == '__main__':
    batch_size = 128
    k_param = 1  # distortion
    alpha = 20
    delta_p = 0.001
    delta_n = 1
    learning_rate = 0.001
    epochs = 10
    n_points = 200

    with open('states_configs_no_rot.pkl', 'rb') as f:
        states_test, configs_test = pickle.load(f)

    config_to_ids = {}
    ids_to_config = {}
    configs_to_color = {}
    color_to_config = {}
    test_ids = np.random.choice(np.arange(configs_test.shape[0]), size=n_points, replace=False)
    for j, c in zip(test_ids, configs_test[test_ids]):
        try:
            config_to_ids[str(c)].append(j)
        except KeyError:
            config_to_ids[str(c)] = [j]
            color = np.random.uniform(0, 1, size=3)
            configs_to_color[str(c)] = color
            color_to_config[str(color)] = str(c)
        ids_to_config[j] = str(c)

    for k in range(1):
        print('Seed {} / 5'.format(k + 1))
        seed = np.random.randint(1e6)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data_anchor, data_positive, data_negative = load_data()
        dataset = TripletDataset(data_anchor, data_positive, data_negative)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        model = GnnModel(state_size=3, output_size=10)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer_info = torch.optim.Adam(model.parameters(), lr=learning_rate)

        logs = defaultdict(list)

        def loss_fn(a, a_positive, a_negative):
            mse_pos = torch.nn.functional.mse_loss(a, a_positive, reduction='sum')
            mse_neg = torch.nn.functional.mse_loss(a, a_negative, reduction='sum')
            return (k_param * torch.nn.functional.relu((-mse_neg + delta_n)) + alpha * torch.nn.functional.relu(mse_pos - delta_p)) / a.size(0)

        for epoch in range(epochs + 1):
            plot_scatter_test(states_test, config_to_ids, configs_to_color)
            for iteration, (states, positives, negatives) in enumerate(data_loader):
                phi_s, phi_sp, phi_sn = model(states, positives, negatives)

                loss = loss_fn(phi_s, phi_sp, phi_sn)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logs['loss'].append(loss.item())
                stop = 1
            print('Epoch {} / {} ---- | Loss = {}'.format(epoch, epochs, loss.item()))

    embeddings = model.network(model.W(torch.Tensor(states_test[test_ids]))).detach().numpy()

    # pca = PCA(n_components=2)
    # pca.fit(embeddings)
    # y = pca.transform(embeddings)

    db = DBSCAN(eps=0.1, min_samples=5).fit(embeddings)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('Found {0} clusters.'.format(n_clusters_))
    stop = 1
    for label in np.unique(labels):
        elements = set()
        ids = np.where(labels == label)[0]
        for id in ids:
            elements.add(str(configs_test[test_ids[id]]))
        print(label, elements)
    # fig = plt.figure()
    # ax = fig.add_subplot()

    # colors = np.array([np.random.uniform(0, 1, size=3) for _ in range(n_clusters_)])
    # aa = colors[labels]
    # ax.scatter(y[:, 0], y[:, 1], c=colors[labels])
    # plt.show()
    # clusters = {}
    # configs = set()
    # for i, embedding in zip(test_ids, embeddings):
    #     if str(ids_to_config[i]) not in configs:
    #         configs.add(str(ids_to_config[i]))
    #         similar = np.where(np.linalg.norm(embeddings - embedding, axis=1) < 0.01)[0]
    #         if len(similar) > 0:
    #             clusters[str(ids_to_config[i])] = []
    #             for s in similar:
    #                 if str(ids_to_config[test_ids[s]]) not in configs:
    #                     configs.add(str(ids_to_config[test_ids[s]]))
    #                     clusters[str(ids_to_config[i])].append(str(ids_to_config[test_ids[s]]))
    #
    # for k, v in clusters.items():
    #     print(k, '|', v)