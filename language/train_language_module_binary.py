import torch
from torch.utils.data import DataLoader
import pickle

from arguments import get_args
from time import ctime
from language.build_dataset import get_dataset
import numpy as np
import os
from language.utils import analyze_inst, Vocab, OneHotEncoder, split_data, get_test_sets
from models import LGG


def launch(args):
    for i in range(args.num_seeds):
        print('\nTraining model for seed {} / {} ...'.format(i + 1, args.num_seeds))
        vocab, configs, device, data_loader, inst_to_one_hot, \
            set_ids, sentences, all_possible_configs, str_to_index = process_data(args)

        train(vocab, configs, device, data_loader, inst_to_one_hot, set_ids,
              sentences, all_possible_configs, str_to_index, args, i)

        args.seed = np.random.randint(1e6)


def process_data(args):
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

    with open(args.save_path + 'inst_to_one_hot.pkl', 'wb') as f:
        pickle.dump(inst_to_one_hot, f)

    with open(args.save_path + 'sentences_list.pkl', 'wb') as f:
        pickle.dump(sorted(set_sentences), f)

    # construct the different test sets
    set_ids, dataset, str_to_index = split_data(configs, sentences, all_possible_configs, all_possible_sentences,
                                                inst_to_one_hot)

    for i, s in enumerate(set_ids):
        print('Len Set ', i, ': ', len(s))

    data_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)

    return vocab, configs, device, data_loader, inst_to_one_hot, set_ids, sentences, all_possible_configs, str_to_index


def train(vocab, configs, device, data_loader, inst_to_one_hot, set_inds, sentences, all_possible_configs, str_to_index, args, vae_id):
    model = LGG(vocab, configs, data_loader, device, args)

    for epoch in range(args.epochs + 1):
        model.train()

    if args.save_model:
        model.save(vae_id)

    results = model.evaluate(configs, sentences, all_possible_configs, set_inds, str_to_index, inst_to_one_hot)

    with open(args.save_path + 'res{}.pkl'.format(vae_id), 'wb') as f:
        pickle.dump(results, f)
    return results.copy()


if __name__ == '__main__':
    # Get parameters
    args = get_args()

    args.save_path = os.path.join(os.getcwd(), args.save_dir)
    print('[{}] Launching Language Goal Generator training'.format(ctime()))
    print('Relational Generator: {}'.format(args.relational))
    print('Conditional Inference: {}'.format(args.conditional_inference))

    launch(args)

