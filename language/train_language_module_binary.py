import os
import time
import torch
import argparse
from torch.utils.data import DataLoader
from collections import defaultdict
from language.utils import analyze_inst, Vocab, OneHotEncoder, ConfigLanguageDataset
from language.vae import ContextVAE
from language.build_dataset import get_dataset
import numpy as np
import pickle
import env
import gym

SAVE_PATH = './data/'
def get_test_sets(configs, sentences, set_inds, all_possible_configs, str_to_index):

    configs = configs[set_inds]
    sentences = np.array(sentences)[set_inds].tolist()

    config_init_and_sentence = []
    for i in range(configs.shape[0]):
        config_init_and_sentence.append(str(configs[i, 0]) + sentences[i])
    unique, idx, idx_in_array = np.unique(np.array(config_init_and_sentence), return_inverse=True, return_index=True)
    train_inits = []  # contains initial configurations
    train_sents = []  # contains sentences
    train_finals_dataset = []  # contains all compatible final configurations found in data
    train_finals_possible = []  # contains all compatible final configurations (oracle)
    for i, i_array in enumerate(idx):
        train_inits.append(configs[i_array, 0])
        train_sents.append(sentences[i_array])
        idx_finals = np.argwhere(idx_in_array == i).flatten()
        init_sent_str = config_init_and_sentence[i_array]
        final_confs = all_possible_configs[str_to_index[init_sent_str], 1]
        final_str = [str(c) for c in final_confs]
        train_finals_possible.append(list(set(final_str)))
        c_f_dataset = [str(c) for c in configs[idx_finals, 1]]
        train_finals_dataset.append(list(set(c_f_dataset)))

    return train_inits, train_sents, train_finals_dataset, train_finals_possible

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    configs, sentences, _, all_possible_configs, all_possible_sentences = get_dataset()


    set_sentences = set(sentences)
    split_instructions, max_seq_length, word_set = analyze_inst(set_sentences)
    vocab = Vocab(word_set)
    one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
    inst_to_one_hot = dict()
    for s_instr in split_instructions:
        inst_to_one_hot[' '.join(s_instr)] = one_hot_encoder.encode(s_instr)


    all_str = ['start' + str(c[0]) + s + str(c[1]) +'end' for c, s in zip(configs, sentences)]
    all_possible_configs_str = [str(c[0]) + s for c, s in zip(all_possible_configs, all_possible_sentences)]

    # Test set 1
    remove1 = [[[0, 1, 0, 0, 0, 0, 0, 0, 0], 'Put blue close_to green'],
               [[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Put green below red']]
    remove1_str = ['start' +  str(np.array(r[0])) + r[1] for r in remove1]

    # Test set 2
    remove2 = [[1, 1, 0, 0, 0, 0, 0, 0, 0]]
    remove2_str = ['start' + str(np.array(r)) for r in remove2]

    # Test set 3
    remove3 = ['Put green on_top_of red', 'Put blue far_from red']
    remove3_str = remove3.copy()

    # Find indices of the different sets in the total dataset.
    set_inds = [[] for _ in range(5)]
    inds_final = []
    for i, s in enumerate(all_str):

        to_remove = False

        used = False

        if not used:
            for s1 in remove1_str:
                if s1 in s:
                    set_inds[1].append(i)
                    used = True
                    break

        if not used:
            for s2 in remove2_str:
                if s2 in s:
                    does_s_also_contains_s_from_r3 = False
                    for s3 in remove3_str:
                        if s2 + s3 in s:
                            does_s_also_contains_s_from_r3 = True
                    used = True
                    if not does_s_also_contains_s_from_r3:
                        set_inds[2].append(i)
                    else:
                        set_inds[4].append(i)
                    break

        if not used:
            for s3 in remove3_str:
                if s3 in s:
                    does_s_also_contains_s_from_r2 = False
                    for s2 in remove2_str:
                        if s2 + s3 in s:
                            does_s_also_contains_s_from_r2 = True
                    used = True
                    if not does_s_also_contains_s_from_r2:
                        set_inds[3].append(i)
                    else:
                        set_inds[4].append(i)
                    break

        if not used and not to_remove:
            set_inds[0].append(i)

    assert np.sum([len(ind) for ind in set_inds]) + len(inds_final) == len(all_str)

    for i, s in enumerate(set_inds):
        print('Len Set ', i, ': ', len(s))

    # dictionary translating string of init config and sentence to all possible final config (id in all_possible_configs)
    # including the ones in the dataset, but also other synthetic ones. This is used for evaluation
    str_to_index = dict()
    for i_s, s in enumerate(all_possible_configs_str):
        if s in str_to_index.keys():
            str_to_index[s].append(i_s)
        else:
            str_to_index[s] = [i_s]
    for k in str_to_index.keys():
        str_to_index[k] = np.array(str_to_index[k])

    train_test_data = get_test_sets(configs, sentences, set_inds[0], all_possible_configs, str_to_index)
    valid_inds = np.array(set_inds[0])
    dataset = ConfigLanguageDataset(configs[valid_inds],
                                    np.array(sentences)[valid_inds].tolist(),
                                    None,
                                    inst_to_one_hot,
                                    binary=False)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)


    return vocab, configs, device, data_loader, inst_to_one_hot, train_test_data, set_inds, sentences, all_possible_configs, str_to_index


def train(vocab, configs, device, data_loader, inst_to_one_hot, train_test_data, set_inds, sentences,
          layers, embedding_size, latent_size, learning_rate, k_param, all_possible_configs, str_to_index, args, vae_id):

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (BCE + k_param * KLD) / x.size(0)


    vae = ContextVAE(vocab.size, inner_sizes=layers, state_size=configs.shape[2], embedding_size=embedding_size, latent_size=latent_size).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    logs = defaultdict(list)

    results = np.zeros([len(set_inds), 8])

    for epoch in range(args.epochs + 1):
        for iteration, (init_state, sentence, state) in enumerate(data_loader):

            init_state, state, sentence = init_state.to(device), state.to(device), sentence.to(device)


            recon_state, mean, log_var, z = vae(init_state, sentence, state)

            target = state
            loss = loss_fn(recon_state, target, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

    with open(SAVE_PATH + 'vae_model{}.pkl'.format(vae_id), 'wb') as f:
        torch.save(vae, f)

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
            c_ii, s_one_hot = torch.Tensor(c_ii).to(device), torch.Tensor(one_hot).to(device)

            x = (vae.inference(c_ii, s_one_hot, n=factor).detach().numpy() > 0.5).astype(np.int)


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
    with open(SAVE_PATH + 'res{}.pkl'.format(vae_id), 'wb') as f:
        pickle.dump(results, f)
    return results.copy()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=np.random.randint(1e6))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    # good ones
    embedding_size = 100
    layers = [128, 128]
    learning_rate = 0.005
    latent_size = 27
    k_param = 0.6

    vocab, configs, device, data_loader, inst_to_one_hot, train_test_data, set_inds, sentences, \
    all_possible_configs, str_to_index = main(args)

    train(vocab, configs, device, data_loader,
          inst_to_one_hot, train_test_data, set_inds, sentences,
          layers, embedding_size, latent_size, learning_rate,  k_param, all_possible_configs, str_to_index, args, 0)
