import numpy as np
import pickle as pkl
import env
import gym


num_samples = 10000
nb_objects = 3
env_name = 'FetchManipulate3Objects-v0'

states = []
configs = []
env = gym.make(env_name)
for i in range(num_samples):
    obs = env.reset_goal(np.zeros([9]), biased_init=True)
    states.append(np.stack([obs['observation'][10 + 15 * j: 13 + 15 * j] for j in range(nb_objects)]))
    configs.append(obs['achieved_goal'])

states = np.array(states)
configs = np.array(configs)
with open('states_configs_no_rot.pkl', 'wb') as f:
    pkl.dump((states, configs), f)
stop = 1