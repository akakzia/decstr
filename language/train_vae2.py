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

def get_test_sets(configs, sentences, set_inds):

    configs = configs[set_inds]
    sentences = np.array(sentences)[set_inds].tolist()

    config_init_and_sentence = []
    for i in range(configs.shape[0]):
        config_init_and_sentence.append(str(configs[i, 0]) + sentences[i])
    unique, idx, idx_in_array = np.unique(np.array(config_init_and_sentence), return_inverse=True, return_index=True)
    train_inits = []
    train_sents = []
    train_finals = []
    for i, i_array in enumerate(idx):
        train_inits.append(configs[i_array, 0])
        train_sents.append(sentences[i_array])
        idx_finals = np.argwhere(idx_in_array == i).flatten()
        train_finals.append(configs[idx_finals, 1])

    return train_inits, train_sents, train_finals

def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    configs, sentences = get_dataset()


    set_sentences = set(sentences)
    split_instructions, max_seq_length, word_set = analyze_inst(set_sentences)
    vocab = Vocab(word_set)
    one_hot_encoder = OneHotEncoder(vocab, max_seq_length)
    inst_to_one_hot = dict()
    for s_instr in split_instructions:
        inst_to_one_hot[' '.join(s_instr)] = one_hot_encoder.encode(s_instr)


    all_str = ['start' + str(c[0]) + s + str(c[1]) + 'end' for c, s in zip(configs, sentences)]

    # test particular combinations of init, sentence, final
    # this tests the extrapolation to different final states than the ones in train set
    remove1 = [[[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Get blue and red far_from each_other', [0, 1, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Put blue above green', [1, 0, 0, 0, 1, 0, 0, 0, 0]],
               [[0, 0, 0, 0, 0, 0, 0, 0, 0], 'Get blue close_to red', [0, 0, 1, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 0, 0, 0, 0, 0, 0, 0], 'Bring red and green together', [1, 1, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Put green on_top_of blue', [1, 0, 0, 1, 0, 0, 0, 0, 0]]]
    remove1_str = ['start' + str(np.array(r[0])) + r[1] + str(np.array(r[2])) for r in remove1]

    remove2 = [[[0, 1, 0, 0, 0, 0, 0, 0, 0], 'Get blue close_to red'],
               [[1, 0, 0, 0, 0, 0, 0, 0, 0], 'Put green above red']]
    remove2_str = ['start' +  str(np.array(r[0])) + r[1] for r in remove2]

    remove3 = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0]]
    remove3_str = ['start' + str(np.array(r)) for r in remove3]

    remove4 = ['Put green on_top_of red', 'Put blue under green', 'Bring red and blue apart']
    remove4_str = remove4.copy()

    # what about removing all of one final state, or combinations of sentence and final state, or init and final ?

    set_inds = [[] for _ in range(6)]
    for i, s in enumerate(all_str):

        to_remove = False

        used = False
        for s1 in remove1_str:
            if s1 in s:
                set_inds[1].append(i)
                used = True
                break

        if not used:
            for s2 in remove2_str:
                if s2 in s:
                    set_inds[2].append(i)
                    used = True
                    break

        if not used:
            for s3 in remove3_str:
                if s3 in s:
                    does_s_also_contains_s_from_r4 = False
                    for s4 in remove4_str:
                        if s3 + s4 in s:
                            does_s_also_contains_s_from_r4 = True
                    used = True
                    if not does_s_also_contains_s_from_r4:
                        set_inds[3].append(i)
                    else:
                        set_inds[5].append(i)
                    break

        if not used:
            for s4 in remove4_str:
                if s4 in s:
                    does_s_also_contains_s_from_r3 = False
                    for s3 in remove3_str:
                        if s3 + s4 in s:
                            does_s_also_contains_s_from_r3 = True
                    used = True
                    if not does_s_also_contains_s_from_r3:
                        set_inds[4].append(i)
                    else:
                        set_inds[5].append(i)
                    break

        if not used and not to_remove:
            set_inds[0].append(i)

    assert np.sum([len(ind) for ind in set_inds]) == len(all_str)

    train_test_data = get_test_sets(configs, sentences, set_inds[0])
    valid_inds = np.array(set_inds[0])
    dataset = ConfigLanguageDataset(configs[valid_inds], np.array(sentences)[valid_inds].tolist(), inst_to_one_hot)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    return vocab, configs, device, data_loader, loss_fn, inst_to_one_hot, train_test_data, set_inds, sentences


def train(vocab, configs, device, data_loader, loss_fn, inst_to_one_hot, train_test_data, set_inds, sentences,
          layers, embedding_size, latent_size, learning_rate, args):
    vae = ContextVAE(vocab.size, inner_sizes=layers, state_size=configs.shape[2], embedding_size=embedding_size, latent_size=latent_size).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        for iteration, (init_state, sentence, state) in enumerate(data_loader):

            init_state, state, sentence = init_state.to(device), state.to(device), sentence.to(device)


            recon_state, mean, log_var, z = vae(init_state, sentence, state)


            loss = loss_fn(recon_state, state, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(data_loader)-1:
                # print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                #     epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))


                score = 0
                for c_i, s, c_f in zip(*train_test_data):
                    one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
                    c_i = np.expand_dims(c_i, 0)
                    c_i, s = torch.Tensor(c_i).to(device), torch.Tensor(one_hot).to(device)
                    x = (vae.inference(c_i, s, n=1).detach().numpy().flatten()>0.5).astype(np.int)
                    c_f_str = []
                    for c in c_f:
                        c_f_str.append(str(c))
                    if str(x) in c_f_str:
                        score += 1
                # print('Score train set: ', score / len(train_test_data[0]))

    stop = 1

    results = np.zeros([len(set_inds), 2])
    # test train statistics
    factor = 50
    for i_gen in range(len(set_inds)):
        if i_gen == 0:
            set_name = 'Train'
        else:
            set_name = 'Test ' + str(i_gen)

        scores = []
        at_least_1 = []
        false_preds = []
        variabilities = []
        data_set = get_test_sets(configs, sentences, set_inds[i_gen])
        for c_i, s, c_f in zip(*data_set):
            one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
            c_i = np.expand_dims(c_i, 0)
            one_hot = np.repeat(one_hot, factor, axis=0)
            c_i = np.repeat(c_i, factor, axis=0)
            c_i, s = torch.Tensor(c_i).to(device), torch.Tensor(one_hot).to(device)

            x = (vae.inference(c_i, s, n=factor).detach().numpy() > 0.5).astype(np.int)
            x_strs = [str(xi) for xi in x]
            variabilities.append(len(set(x_strs)))
            c_f_str = []
            for c in c_f:
                c_f_str.append(str(c))

            count_found = 0
            at_least_1_true = False
            false_preds.append(0)
            for x_str in set(x_strs):
                if x_str in c_f_str:
                    count_found += 1
                    at_least_1_true = True
                else:
                    false_preds[-1] += 1
            scores.append(count_found / len(c_f_str))
            at_least_1.append(at_least_1_true)
            false_preds[-1] /= factor
        print('\n{}: Average of percentage of final states found: {}'.format(set_name, np.mean(scores)))
        print('{}: At least one found: {}'.format(set_name, np.mean(at_least_1)))
        print('{}: Average variability: {}'.format(set_name, np.mean(variabilities)))
        print('{}: Average percentage of false preds: {}'.format(set_name, np.mean(false_preds)))
        results[i_gen, 0] = np.mean(scores)
        results[i_gen, 1] = np.mean(false_preds)

    return results




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    # good ones
    embedding_size = 50
    layers = [64, 64]
    learning_rate = 0.005
    latent_size = 18

    vocab, configs, device, data_loader, loss_fn, inst_to_one_hot, train_test_data, set_inds, sentences = main(args)
    import time
    results = np.zeros([4, 3, 3, 3, 6, 2])
    count = results.size / 12
    counter = 0
    path = '/home/flowers/Desktop/Scratch/sac_curriculum/language/data/'
    for i, embedding_size in enumerate([10, 20, 50, 100]):
        for j, layers in enumerate([[64], [64, 64], [128, 128]]):
            for k, learning_rate in enumerate([0.001, 0.005, 0.01]):
                for l, latent_size in enumerate([9, 18, 27]):
                    t_i = time.time()
                    print('\n', embedding_size, layers, learning_rate, latent_size)
                    results[i, j, k, l, :, :] = train(vocab, configs, device, data_loader, loss_fn,
                                                      inst_to_one_hot, train_test_data, set_inds, sentences,
                                                      layers, embedding_size, latent_size, learning_rate, args)
                    with open(path + 'results.pk', 'wb') as f:
                        pickle.dump(results, f)
                    counter += 1
                    print(counter / count , '%', time.time() - t_i)


