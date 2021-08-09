import numpy as np
import pickle as pkl


def build_dataset(path, contrastive=False):
    """ Given a path of trajectories, builds a contrastive dataset """
    print('getting data from {} ...\n'.format(path))
    with open(path, 'rb') as f:
        trajectories = pkl.load(f)
    n_trajectories = len(trajectories)
    dim_g = len(trajectories[0]['ag'][0])
    n_blocks = int((1 + np.sqrt(1 + (8 * dim_g)/3)) / 2)
    print('number of trajectories in training dataset: {}'.format(n_trajectories))
    print('dimension of goal space: {}'.format(dim_g))
    print('number of blocks to manipulate: {}'.format(n_blocks))

    if contrastive:
        data_anchor = []
        data_positive = []
        data_negative = []
        label_anchor = []
        label_positive = []
        label_negative = []
        for trajectory in trajectories:
            if len(data_anchor) == 175:
                stop = 1
            len_trajectory = len(trajectory['ag'])
            str_trajectory = np.array([str(np.clip(1 + ag, 0, 1)) for ag in trajectory['ag']])
            unique_configs, ids, counts = np.unique(str_trajectory, return_index=True, return_counts=True)
            i = 0
            while i < len(counts):
                running_i = ids[i]
                if str(trajectory['ag'][running_i]) == str(trajectory['ag'][running_i + counts[i] - 1]):
                    label_anchor.append(np.clip(1 + trajectory['ag'][running_i], 0, 1))
                    data_anchor.append(np.concatenate([trajectory['obs'][running_i][10 + 15*k: 13 + 15*k] for k in range(n_blocks)]))
                    label_positive.append(np.clip(1 + trajectory['ag'][running_i + counts[i] - 1], 0, 1))
                    data_positive.append(np.concatenate([trajectory['obs'][running_i][10 + 15*k: 13 + 15*k] for k in range(n_blocks)]))
                    i_neg = np.random.randint(len(counts))
                    c = 0
                    while (i_neg == i or counts[i_neg] < 2) and c < 10:
                        i_neg = np.random.randint(len(counts))
                        c += 1
                    if c == 10:
                        i_neg = (i + 1) % len(counts)
                    label_negative.append(np.clip(1 + trajectory['ag'][np.where(str_trajectory == unique_configs[i_neg])], 0, 1))
                    data_negative.append(np.array([np.concatenate([t[10 + 15*k: 13 + 15*k] for k in range(n_blocks)])
                                          for t in trajectory['obs'][np.where(str_trajectory == unique_configs[i_neg])]]))
                    # data_negative.append(np.concatenate([trajectory['obs'][np.where(str_trajectory == unique_configs[i_neg])][10 + 15*k: 13 + 15*k]
                    #                                      for k in range(n_blocks)]))
                i += 1
        data_anchor = np.array(data_anchor)
        data_positive = np.array(data_positive)

        label_anchor = np.array(label_anchor)
        label_positive = np.array(label_positive)
        assert (label_anchor == label_positive).all()
        with open('data/mutual_info_dataset.pkl', 'wb') as f:
            pkl.dump((data_anchor, data_positive, data_negative, label_anchor, label_positive, label_negative), f)

    else:
        dropped_observations = 0
        data_s = []
        data_next_s = []
        data_trajectories = []
        for k, trajectory in enumerate(trajectories):
            success_id = np.where(trajectory['success'])[0]
            if len(success_id) > 0 and success_id[0] > 0:
                data_s.append(np.concatenate([trajectory['obs'][0][10 + 15*k: 13 + 15*k] for k in range(n_blocks)]))
                data_next_s.append(np.concatenate([trajectory['obs'][success_id[-1]][10 + 15 * k: 13 + 15 * k] for k in range(n_blocks)]))
                data_trajectories.append(trajectory['obs'][:success_id[-1]])
            else:
                dropped_observations += 1
        print('Number of dropped observations: ', dropped_observations)
        data_s = np.array(data_s)
        data_next_s = np.array(data_next_s)

        with open('data/mutual_info_dataset_4.pkl', 'wb') as f:
            pkl.dump((data_s, data_next_s, data_trajectories), f)





if __name__ == '__main__':
    path = 'data/trajectories_5blocks_reduced.pkl'
    build_dataset(path)