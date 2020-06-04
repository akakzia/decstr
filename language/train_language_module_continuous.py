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

SAVE_PATH = "./data/"
def get_test_sets(configs, sentences, set_inds, states, all_possible_configs, str_to_index):

    configs = configs[set_inds]
    states = states[set_inds]
    sentences = np.array(sentences)[set_inds].tolist()

    config_init_and_sentence = []
    for i in range(configs.shape[0]):
        config_init_and_sentence.append(str(configs[i, 0]) + sentences[i])
    unique, idx, idx_in_array = np.unique(np.array(config_init_and_sentence), return_inverse=True, return_index=True)
    train_inits = []
    train_sents = []
    train_finals_dataset = []
    train_finals_possible = []
    train_cont_inits = []
    for i, i_array in enumerate(np.arange(len(sentences))):
        train_inits.append(configs[i_array, 0])
        train_sents.append(sentences[i_array])
        train_cont_inits.append(states[i_array, 0])
        # find all final configs compatible with init and sentence (in dataset and in possible configs)
        init_sent_str = config_init_and_sentence[i_array]
        # find all possible final configs (from dataset + synthetic)
        final_confs = all_possible_configs[str_to_index[init_sent_str], 1]
        final_str = [str(c) for c in final_confs]
        train_finals_possible.append(final_str)
        # find all possible final configs (from dataset)
        id_in_unique = unique.tolist().index(init_sent_str)
        idx_finals = np.argwhere(idx_in_array == id_in_unique).flatten()
        unique_final = np.unique(configs[idx_finals, 1], axis=0)
        c_f_dataset = [str(c) for c in unique_final]
        train_finals_dataset.append(c_f_dataset)
    return train_inits, train_sents, train_finals_dataset, train_finals_possible, train_cont_inits

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    configs, sentences, states, all_possible_configs, all_possible_sentences = get_dataset(binary=False)

    s_min = s_max = None
    set_sentences = set(sentences)
    split_instructions, max_seq_length, word_set = analyze_inst(set_sentences)
    vocab = Vocab(word_set)
    one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
    inst_to_one_hot = dict()
    for s_instr in split_instructions:
        inst_to_one_hot[' '.join(s_instr)] = one_hot_encoder.encode(s_instr)


    all_str = ['start' + str(c[0]) + s + str(c[1]) + 'end' for c, s in zip(configs, sentences)]
    all_possible_configs_str = [str(c[0]) + s for c, s in zip(all_possible_configs, all_possible_sentences)]

    remove1 = [[[0, 1, 0, 0, 0, 0, 0, 0, 0], 'Put blue close_to green'],
               [[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Put green below red']]
    remove1_str = ['start' +  str(np.array(r[0])) + r[1] for r in remove1]

    remove2 = [[1, 1, 0, 0, 0, 0, 0, 0, 0]]
    remove2_str = ['start' + str(np.array(r)) for r in remove2]
    remove3 = ['Put green on_top_of red', 'Put blue far_from red']
    remove3_str = remove3.copy()


    set_inds = [[] for _ in range(5)]
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

    assert np.sum([len(ind) for ind in set_inds]) == len(all_str)

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
    for i, s in enumerate(set_inds):
        print('Set ', i, ': ', len(s))
    train_test_data = get_test_sets(configs, sentences, set_inds[0], states, all_possible_configs, str_to_index)
    test_data = [get_test_sets(configs, sentences, set_ids, states, all_possible_configs, str_to_index) for set_ids in set_inds[1:]]
    valid_inds = np.array(set_inds[0])
    dataset = ConfigLanguageDataset(configs[valid_inds],
                                    np.array(sentences)[valid_inds].tolist(),
                                    states[valid_inds],
                                    inst_to_one_hot,
                                    binary=False)
    configs = None
    sentences = None
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)


    return   vocab, configs, device, data_loader, inst_to_one_hot, \
    train_test_data, test_data, set_inds, sentences, all_possible_configs, str_to_index , states, s_min, s_max



def train(vocab, configs, states, device, data_loader, inst_to_one_hot, train_test_data, test_data, set_inds,sentences,
              layers, embedding_size, latent_size, learning_rate,s_min, s_max,k_param, all_possible_configs, str_to_index, args, VAE_ID):

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        print(BCE, KLD)
        return (BCE + k_param * KLD) / x.size(0)

    def loss_fn_soft_bce(recon_x, x, mean, log_var):
        recon_x = torch.clamp(recon_x, min=1e-4, max=1 - 1e-4)
        log_norm_const = cont_bern_log_norm(recon_x)
        log_p_all = torch.sum(x * torch.log(recon_x) + (1 - x) * torch.log(1 - recon_x) + log_norm_const, dim=1)
        log_p = torch.mean(log_p_all)

        KL = 0.5 * torch.sum(mean.pow(2) + log_var.exp() - log_var - 1.0, dim=1)
        KL = torch.mean(KL)

        cost = - log_p + k_param * KL
        return cost

    def atanh(x):
        return 0.5 * torch.log(1 + x) - torch.log(1 - x)

    def cont_bern_log_norm(lam, l_lim=0.49, u_lim=0.51):
        # computes the log normalizing constant of a continuous Bernoulli distribution in a numerically stable way.
        # returns the log normalizing constant for lam in (0, l_lim) U (u_lim, 1) and a Taylor approximation in
        # [l_lim, u_lim].
        # cut_y below might appear useless, but it is important to not evaluate log_norm near 0.5 as tf.where evaluates
        # both options, regardless of the value of the condition.
        cut_lam = torch.where((lam < l_lim) + (lam > u_lim), lam, l_lim * torch.ones(lam.shape))
        log_norm = torch.log(torch.abs(2.0 * atanh(1 - 2.0 * cut_lam))) - torch.log(torch.abs(1 - 2.0 * cut_lam))
        taylor = np.log(2.0) + 4.0 / 3.0 * torch.pow(lam - 0.5, 2) + 104.0 / 45.0 * torch.pow(lam - 0.5, 4)
        return torch.where((lam < l_lim) + (lam > u_lim), log_norm, taylor)



    def loss_fn_cont(recon_x, x, mean, log_var):
        MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return (MSE + k_param * KLD) / x.size(0)

    loss_fn = loss_fn_cont

    vae = ContextVAE(vocab.size, binary=False, inner_sizes=layers, state_size=states.shape[2], embedding_size=embedding_size, latent_size=latent_size).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    logs = defaultdict(list)
    env = gym.make('FetchManipulate3ObjectsContinuous-v0')
    get_config = env.unwrapped._get_configuration
    results = np.zeros([len(set_inds), 8])

    for epoch in range(args.epochs + 1):

        for iteration, (init_config, sentence, config, init_state, state) in enumerate(data_loader):

            init_state, state, sentence = init_state.to(device), state.to(device), sentence.to(device)

            recon_state, mean, log_var, z = vae(init_state, sentence, state)

            loss = loss_fn(recon_state, state, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

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
        if i_gen == 0:
            data_set = train_test_data
        else:
            data_set = test_data[i_gen - 1]
        for c_i, s, c_f_dataset, c_f_possible, co_i in zip(*data_set):
            c_f_possible = set(c_f_possible)
            c_f_dataset = set(c_f_dataset)
            count += 1
            one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
            co_i = np.expand_dims(co_i, 0)
            one_hot = np.repeat(one_hot, factor, axis=0)
            co_i = np.repeat(co_i, factor, axis=0)
            co_i, s = torch.Tensor(co_i).to(device), torch.Tensor(one_hot).to(device)

            x = vae.inference(co_i, s, n=factor).detach().numpy()
            x_strs = []
            for xi in x:
                xi = get_config(xi.reshape([3, 3])).astype(np.int)
                x_strs.append(str(xi))
            variabilities.append(len(set(x_strs)))
            count_found_dataset = 0
            count_found_possible = 0
            count_found_not_dataset = 0
            count_false_pred = 0
            # count coverage of final configs in dataset
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
    with open(SAVE_PATH + 'continuous_res{}.pkl'.format(VAE_ID), 'wb') as f:
        pickle.dump(results, f)
    return results.copy()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=np.random.randint(1e6))
    parser.add_argument("--epochs", type=int, default=150)
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

    vocab, configs, device, data_loader, inst_to_one_hot, \
    train_test_data, test_data, set_inds, sentences, all_possible_configs, str_to_index , states, s_min, s_max = main(args)


    train(vocab, configs, states, device, data_loader, inst_to_one_hot, train_test_data, test_data, set_inds,sentences,
          layers, embedding_size, latent_size, learning_rate,s_min, s_max,k_param, all_possible_configs, str_to_index, args, 0)

