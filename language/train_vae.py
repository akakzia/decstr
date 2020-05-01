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


    all_str = [str(c[0]) + s + str(c[1]) for c, s in zip(configs, sentences)]
    config_init_and_sentence = []
    for i in range(configs.shape[0]):
        config_init_and_sentence.append(str(configs[i, 0]) + sentences[i])
    unique, idx, idx_in_array = np.unique(np.array(config_init_and_sentence), return_inverse=True, return_index=True)

    # test particular combinations of init, sentence, final
    # this tests the extrapolation to different final states than the ones in train set
    remove1 = [[[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Get blue and red far_from each_other', [0, 1, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Put blue above green', [1, 0, 0, 0, 1, 0, 0, 0, 0]],
               [[0, 0, 0, 0, 0, 0, 0, 0, 0], 'Get blue close_to red', [0, 0, 1, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 0, 0, 0, 0, 0, 0, 0], 'Bring red and green together', [1, 1, 0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 1, 0, 0, 0, 0, 0, 0], 'Put green on_top_of blue', [1, 0, 0, 1, 0, 0, 0, 0, 0]]]
    test1_inits = [np.array(r[0]) for r in remove1]
    test1_sents = [r[1] for r in remove1]
    test1_finals = [[np.array(r[2])] for r in remove1]
    str_search = [str(np.array(r[0])) + r[1] for r in remove1]
    idx_test1 = [unique.tolist().index(str_i) for str_i in str_search]
    str_to_remove = [str(np.array(r[0])) + r[1] + str(np.array(r[2])) for r in remove1]


    # test particular combinations of init, sentence
    remove2 = [[[0, 1, 0, 0, 0, 0, 0, 0, 0], 'Get blue close_to red'], [[1, 0, 0, 0, 0, 0, 0, 0, 0], 'Put green above red']]
    test2_inits = [np.array(r[0]) for r in remove2]
    test2_sents = [r[1] for r in remove2]
    test2_finals = []
    str_search = [str(np.array(r[0])) + r[1] for r in remove2]
    idx_to_remove = [unique.tolist().index(str_i) for str_i in str_search]
    assert len(idx_to_remove) == len(remove2)
    for i in idx_to_remove:
        idx_finals = np.argwhere(idx_in_array == i).flatten()
        test2_finals.append(configs[idx_finals, 1])

    for i, s, fs in zip(test2_inits, test2_sents, test2_finals):
        for f in fs:
            str_to_remove.append(str(i) + s + str(f))


    # test particular init
    remove3 = [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0]]
    remove3_str = [str(np.array(r)) for r in remove3]
    remove4 = ['Put green on_top_of red', 'Put blue under green', 'Bring red and blue apart']
    idx_inits = []
    test3_inits = []
    test3_sents = []
    test3_finals = []
    for i in range(len(remove3)):
        idx_inits.append([])
        for j, str_unique in enumerate(unique):
            if str(np.array(remove3[i])) in str_unique:
                if sentences[idx[j]] in remove4:
                    idx_to_remove.append(j)
                else:
                    idx_inits[-1].append(j)
                    idx_to_remove.append(j)
                    test3_inits.append(configs[idx[j], 0])
                    test3_sents.append(sentences[idx[j]])
                    idx_finals = np.argwhere(idx_in_array == j).flatten()
                    test3_finals.append(configs[idx_finals, 1])

    for i, s, fs in zip(test3_inits, test3_sents, test3_finals):
        for f in fs:
            str_to_remove.append(str(i) + s + str(f))

    # test particular sentence: synonym understanding
    remove4 = ['Put green on_top_of red', 'Put blue under green', 'Bring red and blue apart']
    idx_sents = []
    test4_inits = []
    test4_sents = []
    test4_finals = []
    for i in range(len(remove4)):
        idx_sents.append([])
        for j, str_unique in enumerate(unique):
            if remove4[i] in str_unique:
                if str(np.array(configs[idx[j], 0])) in remove3_str:
                    idx_to_remove.append(j)
                else:
                    idx_sents[-1].append(j)
                    idx_to_remove.append(j)
                    test4_inits.append(configs[idx[j], 0])
                    test4_sents.append(sentences[idx[j]])
                    idx_finals = np.argwhere(idx_in_array == j).flatten()
                    test4_finals.append(configs[idx_finals, 1])

    for i, s, fs in zip(test4_inits, test4_sents, test4_finals):
        for f in fs:
            str_to_remove.append(str(i) + s + str(f))

    train_inits = []
    train_sents = []
    train_finals = []
    for i, i_array in enumerate(idx):
        if i not in idx_to_remove:
            if i in idx_test1:
                id_which = idx_test1.index(i)
                train_inits.append(configs[i_array, 0])
                train_sents.append(sentences[i_array])
                idx_finals = np.argwhere(idx_in_array == i).flatten()
                train_finals.append(configs[idx_finals, 1])
                str_configs = [str(c) for c in train_finals[-1]]
                valid_ids = []
                for id in range(len(str_configs)):
                    if str_configs[id] != str(np.array(remove1[id_which][2])):
                        valid_ids.append(id)
                train_finals[-1] = train_finals[-1][np.array(valid_ids)]
            else:
                train_inits.append(configs[i_array, 0])
                train_sents.append(sentences[i_array])
                idx_finals = np.argwhere(idx_in_array == i).flatten()
                train_finals.append(configs[idx_finals, 1])


    valid_idx_in_train_set = []
    count = 0
    for i, s in enumerate(all_str):
        if s not in str_to_remove:
            valid_idx_in_train_set.append(i)
        else:
            count += 1
    assert count == len(set(str_to_remove))
    valid_idx_in_train_set = np.array(valid_idx_in_train_set)
    configs = configs[valid_idx_in_train_set]
    sentences = np.array(sentences)[valid_idx_in_train_set].tolist()




    dataset = ConfigLanguageDataset(configs, sentences, inst_to_one_hot)
    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (BCE + KLD) / x.size(0)

    vae = ContextVAE(vocab.size, inner_sizes=[64, 64], state_size=configs.shape[2], embedding_size=50, latent_size=configs.shape[2]*2).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

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
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, iteration, len(data_loader)-1, loss.item()))


                score = 0
                for c_i, s, c_f in zip(train_inits, train_sents, train_finals):
                    one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
                    c_i = np.expand_dims(c_i, 0)
                    c_i, s = torch.Tensor(c_i).to(device), torch.Tensor(one_hot).to(device)
                    x = (vae.inference(c_i, s, n=1).detach().numpy().flatten()>0.5).astype(np.int)
                    c_f_str = []
                    for c in c_f:
                        c_f_str.append(str(c))
                    if str(x) in c_f_str:
                        score += 1
                print('Score train set: ', score / len(train_inits))

    stop = 1

    # test train statistics
    factor = 100
    scores = []
    at_least_1 = []
    false_preds = []
    variabilities = []
    for c_i, s, c_f in zip(train_inits, train_sents, train_finals):
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
    print('\nTrain: Average of percentage of final states found: ', np.mean(scores))
    print('Train: At least one found: ', np.mean(at_least_1))
    print('Train: Average variability: ', np.mean(variabilities))
    print('Train: Average percentage of false preds: ', np.mean(false_preds))

    # test generalization 1
    scores = []
    at_least_1 = []
    false_preds = []
    for c_i, s, c_f in zip(test1_inits, test1_sents, test1_finals):
        one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
        c_i = np.expand_dims(c_i, 0)
        one_hot = np.repeat(one_hot, factor, axis=0)
        c_i = np.repeat(c_i, factor, axis=0)
        c_i, s = torch.Tensor(c_i).to(device), torch.Tensor(one_hot).to(device)

        x = (vae.inference(c_i, s, n=factor).detach().numpy() > 0.5).astype(np.int)
        x_strs = [str(xi) for xi in x]

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
    print('\nTest1: Average of percentage of final states found:', np.mean(scores))
    print('Test1: Probability to find at least one of them: ', np.mean(at_least_1))
    print('Test1: Average percentage of false preds: ', np.mean(false_preds))

    # test generalization 2
    scores = []
    at_least_1 = []
    false_preds = []
    for c_i, s, c_f in zip(test2_inits, test2_sents, test2_finals):
        one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
        c_i = np.expand_dims(c_i, 0)
        one_hot = np.repeat(one_hot, factor, axis=0)
        c_i = np.repeat(c_i, factor, axis=0)
        c_i, s = torch.Tensor(c_i).to(device), torch.Tensor(one_hot).to(device)

        x = (vae.inference(c_i, s, n=factor).detach().numpy() > 0.5).astype(np.int)
        x_strs = [str(xi) for xi in x]

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
    print('\nTest2: Average of percentage of final states found:', np.mean(scores))
    print('Test2: Probability to find at least one of them: ', np.mean(at_least_1))
    print('Test2: Average percentage of false preds: ', np.mean(false_preds))

    # test generalization 3
    scores = []
    at_least_1 = []
    false_preds = []
    for c_i, s, c_f in zip(test3_inits, test3_sents, test3_finals):
        one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
        c_i = np.expand_dims(c_i, 0)
        one_hot = np.repeat(one_hot, factor, axis=0)
        c_i = np.repeat(c_i, factor, axis=0)
        c_i, s = torch.Tensor(c_i).to(device), torch.Tensor(one_hot).to(device)

        x = (vae.inference(c_i, s, n=factor).detach().numpy() > 0.5).astype(np.int)
        x_strs = [str(xi) for xi in x]

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
    print('\nTest3: Average of percentage of final states found:', np.mean(scores))
    print('Test3: Probability to find at least one of them: ', np.mean(at_least_1))
    print('Test3: Average percentage of false preds: ', np.mean(false_preds))

    # test generalization 4
    scores = []
    at_least_1 = []
    false_preds = []
    for c_i, s, c_f in zip(test4_inits, test4_sents, test4_finals):
        one_hot = np.expand_dims(np.array(inst_to_one_hot[s.lower()]), 0)
        c_i = np.expand_dims(c_i, 0)
        one_hot = np.repeat(one_hot, factor, axis=0)
        c_i = np.repeat(c_i, factor, axis=0)
        c_i, s = torch.Tensor(c_i).to(device), torch.Tensor(one_hot).to(device)

        x = (vae.inference(c_i, s, n=factor).detach().numpy() > 0.5).astype(np.int)
        x_strs = [str(xi) for xi in x]

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
    print('\nTest4: Average of percentage of final states found:', np.mean(scores))
    print('Test4: Probability to find at least one of them: ', np.mean(at_least_1))
    print('Test4: Average percentage of false preds: ', np.mean(false_preds))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[784, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[256, 784])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true')

    args = parser.parse_args()

    main(args)