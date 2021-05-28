import torch
from gvae import ContextVAE
from collections import defaultdict
import numpy as np
from language.utils import get_test_sets


class LGG:
    def __init__(self, vocabulary, configurations, data_loader,  device, args):
        self.vocabulary_size = vocabulary.size
        self.encoder_layers = args.encoder_layer_sizes
        self.decoder_layers = args.decoder_layer_sizes
        self.state_size = configurations.shape[2]
        self.embedding_size = args.embedding_size
        self.latent_size = args.latent_size
        self.data_loader = data_loader
        self.device = device
        self.conditional_inference = args.conditional_inference

        self.vae = ContextVAE(self.vocabulary_size, encoder_inner_sizes=self.encoder_layers, decoder_inner_sizes=self.decoder_layers,
                              state_size=self.state_size, embedding_size=self.embedding_size, latent_size=self.latent_size,
                              relational=args.relational).to(self.device)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=args.learning_rate)

        def loss_fn(recon_x, x, mean, log_var):
            bce = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
            kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return (bce + args.k_param * kld) / x.size(0)

        self.loss_fn = loss_fn

        self.save_path = args.save_path
        self.colors_to_ids = {'red': 0, 'green': 1, 'blue': 2}

    def train(self):
        logs = defaultdict(list)

        for iteration, (init_state, sentence, state) in enumerate(self.data_loader):
            init_state, state, sentence = init_state.to(self.device), state.to(self.device), sentence.to(self.device)

            recon_state, mean, log_var, z = self.vae(init_state, sentence, state)

            target = state
            loss = self.loss_fn(recon_state, target, mean, log_var)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            logs['loss'].append(loss.item())

    def save(self, vae_id):
        with open(self.save_path + 'vae_model{}.pkl'.format(vae_id), 'wb') as f:
            torch.save(self.vae, f)

    def evaluate(self, configs, sentences, all_possible_configs, set_inds, str_to_index, inst_to_one_hot):
        results = np.zeros([len(set_inds), 8])

        # test train statistics
        factor = 100
        for i_gen in range(len(set_inds)):
            if i_gen == 0:
                set_name = 'Train'
            else:
                set_name = 'Test ' + str(i_gen)

            coverage_dataset = []
            coverage_possible = []
            count = 0
            false_preds = []
            variabilities = []
            nb_cf_possible = []
            nb_cf_dataset = []
            found_beyond_dataset = []
            valid_goals = []
            data_set = get_test_sets(configs, sentences, set_inds[i_gen], all_possible_configs, str_to_index)
            for c_i, s, c_f_dataset, c_f_possible in zip(*data_set):
                c_f_possible = set(c_f_possible)
                c_f_dataset = set(c_f_dataset)
                count += 1
                one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
                c_ii = np.expand_dims(c_i, 0)
                one_hot = np.repeat(one_hot, factor, axis=0)
                c_ii = np.repeat(c_ii, factor, axis=0)
                c_ii, s_one_hot = torch.Tensor(c_ii).to(self.device), torch.Tensor(one_hot).to(self.device)

                if self.conditional_inference:
                    words = set(s.split(' '))
                    colors = set(self.colors_to_ids.keys())
                    l = list(words.intersection(colors))
                    if len(l) == 2:
                        p = (self.colors_to_ids[l[0]], self.colors_to_ids[l[1]])
                    else:
                        p = None
                    x = (self.vae.inference(c_ii, s_one_hot, pair=p, n=factor).detach().numpy() > 0.5).astype(np.int)
                else:
                    x = (self.vae.inference(c_ii, s_one_hot, n=factor).detach().numpy() > 0.5).astype(np.int)

                x_strs = [str(xi) for xi in x]
                variabilities.append(len(set(x_strs)))
                count_found_dataset = 0
                count_found_possible = 0
                count_found_not_dataset = 0
                count_false_pred = 0

                for x_str in set(x_strs):
                    if x_str in c_f_possible:
                        count_found_possible += 1
                        if x_str in c_f_dataset:
                            count_found_dataset += 1
                        else:
                            count_found_not_dataset += 1

                # count false positives, final configs that are not compatible
                for x_str in x_strs:
                    if x_str not in c_f_possible:
                        count_false_pred += 1
                        # print(c_i, s, x_str)

                coverage_dataset.append(count_found_dataset / len(c_f_dataset))
                coverage_possible.append(count_found_possible / len(c_f_possible))
                found_beyond_dataset.append(count_found_not_dataset)
                valid_goals.append(count_found_possible)
                false_preds.append(count_false_pred / factor)
                nb_cf_possible.append(len(c_f_possible))
                nb_cf_dataset.append(len(c_f_dataset))
            print('\n{}: Probability that a sampled goal is valid {}'.format(set_name, 1 - np.mean(false_preds)))
            print('{}: Number of different valid sampled goals: {}'.format(set_name, np.mean(valid_goals)))
            print('{}: Number of valid sampled goals not in dataset: {}'.format(set_name, np.mean(found_beyond_dataset)))
            print('{}: Number of valid goals (oracle): {}'.format(set_name, np.mean(nb_cf_possible)))
            print('{}: Number of valid goals found in dataset: {}'.format(set_name, np.mean(nb_cf_dataset)))
            print('{}: Coverage of all valid goals: {}'.format(set_name, np.mean(coverage_possible)))
            print('{}: Coverage of all valid goals from dataset: {}'.format(set_name, np.mean(coverage_dataset)))
            results[i_gen, 0] = count
            results[i_gen, 1] = 1 - np.mean(false_preds)
            results[i_gen, 2] = np.mean(valid_goals)
            results[i_gen, 3] = np.mean(found_beyond_dataset)
            results[i_gen, 4] = np.mean(nb_cf_possible)
            results[i_gen, 5] = np.mean(nb_cf_dataset)
            results[i_gen, 6] = np.mean(coverage_dataset)
            results[i_gen, 7] = np.mean(coverage_possible)

        return results
