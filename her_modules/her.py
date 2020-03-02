import numpy as np
from scipy.linalg import block_diag


class her_sampler:
    def __init__(self, replay_strategy, replay_k, mc_her, reward_func=None):
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        self.mc_her = mc_her
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def get_multi_criteria(self, future_goals, actual_goals):
        number_blocks = future_goals.shape[1]
        proba = np.random.uniform(0, 1, number_blocks * future_goals.shape[0]).reshape(future_goals.shape[0],
                                                                                       number_blocks)
        indexes = np.where(proba < 0.5, 1, 0)
        res = future_goals*indexes + actual_goals*(1-indexes)
        return res

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1]
        rollout_batch_size = episode_batch['actions'].shape[0]
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = np.random.randint(T, size=batch_size)
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        # her idx
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        if not self.mc_her:
            transitions['g'][her_indexes] = future_ag
        else:
            transitions['g'][her_indexes] = self.get_multi_criteria(future_ag, transitions['g'][her_indexes])
        # to get the params to re-compute reward
        # transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions['r'] = np.expand_dims(np.array([self.reward_func(ag_next, g, None) for ag_next, g in zip(transitions['ag_next'],
                                                                                        transitions['g'])]), 1)
        # Filtered-HER
        """filter_indexes = np.arange(batch_size)
        delete_indexes = np.array([])
        for i, reward in enumerate(transitions['r'][1:]):
            if (transitions['g'][i] == transitions['g'][i+1]).all():
                if transitions['r'][i] > 0.:
                    batch_size -= 1
                    delete_indexes = np.concatenate((delete_indexes, [i+1]))
        filter_indexes = np.delete(filter_indexes, delete_indexes)
        transitions = {k: transitions[k][filter_indexes].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}"""

        return transitions
