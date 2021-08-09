import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle
from utils import TripletDataset, TrajectoriesDataset
from collections import defaultdict
# from models import SimpleModel, GnnModel
# from graph_models import GnnModel
from mutual_models import MutualModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import get_configuration


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)

def load_data(path):
    """
    Load the dataset of configurations and geometric states
    """
    path_states_configs = path
    with open(path_states_configs, 'rb') as f:
        data = pickle.load(f)

    return data

def plot_scatter_test(states, next_states, test_configs, traj_test):
    model.eval()
    pca = PCA(n_components=2)
    # out = model.network(torch.Tensor(states)).detach().numpy()
    out = model.scene_encoder(torch.Tensor(np.concatenate([states, next_states]))).detach().numpy()
    pca.fit(out)
    fig = plt.figure()
    ax = fig.add_subplot()
    cmap = ['red', 'blue', 'green', 'black', 'yellow', 'indigo', 'navy', 'orange', 'pink', 'magenta', 'peru', 'tomato', 'olive',
            'chocolate', 'khaki', 'darkred', 'darkorange']
    for i, c in enumerate(config_to_ids.keys()):
        input_state = torch.Tensor(next_states[config_to_ids[str(c)]])
        embeddings = (model.scene_encoder(input_state).detach().numpy())
        y = pca.transform(embeddings)
        ax.scatter(y[:, 0], y[:, 1], color=cmap[i], label=c)
    plt.legend(title='legend', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.savefig('images_bis/{}.jpg'.format(epoch), bbox_inches='tight')
    plt.close('all')
    model.train()


if __name__ == '__main__':
    batch_size = 128
    k_param = 1  # distortion
    alpha = 20
    delta_p = 0.001
    delta_n = 1
    learning_rate = 0.03
    epochs = 50
    n_points = 200
    lambda_error = 0.
    lambda_info = 0.

    # with open('data/states_configs_no_rot.pkl', 'rb') as f:
    #     states_test, configs_test = pickle.load(f)
    # config_to_ids = {}
    # ids_to_config = {}
    # configs_to_color = {}
    # color_to_config = {}
    # test_ids = np.random.choice(np.arange(configs_test.shape[0]), size=configs_test.shape[0], replace=False)
    # for j, c in zip(test_ids, configs_test[test_ids]):
    #     try:
    #         config_to_ids[str(c)].append(j)
    #     except KeyError:
    #         config_to_ids[str(c)] = [j]
    #         color = np.random.uniform(0, 1, size=3)
    #         configs_to_color[str(c)] = color
    #         color_to_config[str(color)] = str(c)
    #     ids_to_config[j] = str(c)

    path = 'data/mutual_info_dataset_4.pkl'

    # states_anchor, states_positive, states_negative, config_anchor, config_positive, config_negative = load_data(path)
    train_states, train_next_states, trajectories = load_data(path)
    ids = np.random.choice(np.arange(len(train_states)), size=900, replace=False)
    # dataset = TripletDataset(states_anchor, states_positive, states_negative)
    trajectory_test = trajectories[-15]
    dataset = TrajectoriesDataset(train_states[ids], train_next_states[ids], [trajectories[i] for i in ids])

    test_ids = np.random.choice(np.arange(len(train_states)), size=200, replace=False)
    test_configs = np.array([get_configuration(e) for e in train_next_states[test_ids]])

    config_to_ids = {}
    ids_to_config = {}
    configs_to_color = {}
    color_to_config = {}

    for j, c in zip(test_ids, test_configs):
        c = np.clip(1 + c, 0, 1)
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
        print(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        # model = GnnModel(state_size=6, output_size=1)
        model = MutualModel(state_size=10, inner_sizes=[64], objects_size=15, embedding_size=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer_info = torch.optim.Adam(model.parameters(), lr=learning_rate)

        logs = defaultdict(list)
        def mutual_loss(s, next_s, traj, mutual_traj, mutual_states, error):
            loss = torch.nn.functional.mse_loss(next_s, s + traj, reduction='sum') / s.size(0)
            loss_error = torch.linalg.norm(error)
            return loss
            # return loss + lambda_error * loss_error + lambda_info * torch.mean(mutual_traj + mutual_states)

        def loss_fn(a, a_positive, a_negative):
            bce_pos = torch.nn.functional.mse_loss(a, a_positive, reduction='sum')
            bce_neg = torch.nn.functional.mse_loss(a, a_negative, reduction='sum')
            return (bce_pos - bce_neg) / a.size(0)
            # mse_pos = torch.nn.functional.mse_loss(a, a_positive, reduction='sum')
            # mse_neg = torch.nn.functional.mse_loss(a, a_negative, reduction='sum')
            # return (k_param * torch.nn.functional.relu((-mse_neg + delta_n)) + alpha * torch.nn.functional.relu(mse_pos - delta_p)) / a.size(0)
            # return (-mse_neg + mse_pos) / a.size(0)
        for epoch in range(epochs + 1):
            for iteration, (states, next_states, trajectories) in enumerate(data_loader):
            # states, positives, negatives = dataset.sample_data(batch_size)
            #     phi_s, phi_sp, phi_sn = model(states, positives, negatives)
                states, next_states, trajectories = dataset.sample_data(batch_size)
                s_encodings, next_s_encoding, embeddings, \
                half_embeddings, half_s_encodings = model.forward(states, next_states, trajectories)

                mutual_info_trajectory, mutual_info_states = model.compute_mutual_info(s_encodings, next_s_encoding, embeddings,
                                                                                       half_embeddings, half_s_encodings)

                linear_error = model.compute_error(states, trajectories)

                # loss = loss_fn(phi_s, phi_sp, phi_sn)
                loss = mutual_loss(s_encodings, next_s_encoding, embeddings, mutual_info_trajectory,
                                   mutual_info_states, linear_error)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logs['loss'].append(loss.item())
                stop = 1
            print('Update {} / {} ---- | Loss = {}'.format(epoch, epochs, loss.item()))
            plot_scatter_test(train_states, train_next_states, test_configs, trajectory_test)


    # the_states = states_test[test_ids]
    # the_configs = configs_test[test_ids]
    # embeddings = (model.scene_encoder(torch.Tensor(the_states).view(-1, 9)).detach().numpy() > 0.5).astype(np.int)
    # discovered_classes = {str(e) for e in embeddings}
    # for cl in list(discovered_classes):
    #     print('class {}'.format(cl))
    #     ids = np.where(np.array([str(e) == cl for e in embeddings]))[0]
    #     print({str(the_configs[i]) for i in ids})
    # stop = 1
    #
    # pca = PCA(n_components=2)
    # pca.fit(embeddings)
    # y = pca.transform(embeddings)
    #
    # db = DBSCAN(eps=0.1, min_samples=20).fit(embeddings)
    # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    # labels = db.labels_
    #
    # # Number of clusters in labels, ignoring noise if present.
    # n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print('Found {0} clusters.'.format(n_clusters_))
    # stop = 1
    # for label in np.unique(labels):
    #     elements = set()
    #     ids = np.where(labels == label)[0]
    #     for id in ids:
    #         elements.add(str(configs_test[test_ids[id]]))
    #     print(label, elements)
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