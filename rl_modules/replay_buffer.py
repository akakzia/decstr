import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""


class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        # thread lock
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
    
    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx


class MultiHeadBuffer:
    def __init__(self, env_params, buffer_size, heads, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.heads = heads
        self.size = buffer_size // self.T
        # memory management
        self.current_size = [0 for _ in range(heads)]
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = [{'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                         'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                         'g': np.empty([self.size, self.T, self.env_params['goal']]),
                         'actions': np.empty([self.size, self.T, self.env_params['action']]),
                         } for _ in range(self.heads)]
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch, head):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(head, inc=batch_size)
            # store the informations
            self.buffers[head]['obs'][idxs] = mb_obs
            self.buffers[head]['ag'][idxs] = mb_ag
            self.buffers[head]['g'][idxs] = mb_g
            self.buffers[head]['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size

    # sample the data from the replay buffer
    def sample(self, batch_size, head):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers[head].keys():
                temp_buffers[key] = self.buffers[head][key][:self.current_size[head]]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, head, inc=None):
        inc = inc or 1
        if self.current_size[head] + inc <= self.size:
            idx = np.arange(self.current_size[head], self.current_size[head] + inc)
        elif self.current_size[head] < self.size:
            overflow = inc - (self.size - self.current_size[head])
            idx_a = np.arange(self.current_size[head], self.size)
            idx_b = np.random.randint(0, self.current_size[head], overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size[head] = min(self.size, self.current_size[head] + inc)
        if inc == 1:
            idx = idx[0]
        return idx


class MultiBuffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.n_transitions_stored = 0
        self.sample_func = sample_func

        self.current_size = 0
        # create the buffer to store info
        self.buffer = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                         'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                         'g': np.empty([self.size, self.T, self.env_params['goal']]),
                         'actions': np.empty([self.size, self.T, self.env_params['action']]),
                         }
        self.goal_ids = np.zeros([self.size])  # contains id of achieved goal (discovery rank)
        self.goal_ids.fill(np.nan)

        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch, goal_ids):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffer['obs'][idxs] = mb_obs
            self.buffer['ag'][idxs] = mb_ag
            self.buffer['g'][idxs] = mb_g
            self.buffer['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size
            self.goal_ids[idxs] = goal_ids

    # sample the data from the replay buffer
    def sample(self, batch_size, goal_ids_of_head):
        temp_buffers = {}
        with self.lock:
            inds = []
            for g_id in goal_ids_of_head:
                inds += np.argwhere(self.goal_ids == g_id).flatten().tolist()
            inds = np.array(inds)
            for key in self.buffer.keys():
                temp_buffers[key] = self.buffer[key][inds]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = np.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
