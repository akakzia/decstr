from language.get_data import get_data
import numpy as np
from language.rbm import RBM

unique_reached_config_transitions, reached_config_transitions, predicates, \
predicate_to_id, id_to_predicate, colors = get_data()


all_configs = np.concatenate([reached_config_transitions[:,0,:], reached_config_transitions[:,1,:]], axis=0)

dim = all_configs.shape[1]
model = RBM(n_hidden=10, m_observe=dim)

model.train(all_configs, epochs=10)

strings = []
for c in all_configs:
    c = str(c.astype(np.int))
    strings.append(c)
set_string = set(strings)
samples = []
for _ in range(20):
    samples.append(model.sample().flatten())
samples = np.array(samples)

strings_samples = []
for c in samples:
    c = str(c.astype(np.int))
    strings_samples.append(c)

for c in strings_samples:
    if c not in set_string:
        print(c)

stop = 1