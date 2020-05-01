import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch

class ConfigLanguageDataset(Dataset):
    def __init__(self, configs, sentences, string_to_1hot, shuffle=True):

        assert configs.shape[0] == len(sentences)

        one_hots = []
        for s in sentences:
            one_hots.append(string_to_1hot[s.lower()])

        inds = np.arange(configs.shape[0])
        np.random.shuffle(inds)

        self.sentences = np.array(one_hots)[inds].astype(np.float32)
        self.configs = configs[inds].astype(np.float32)


    def __getitem__(self, index):
        return self.configs[index][0], self.sentences[index], self.configs[index][1]

    def __len__(self):
        return self.configs.shape[0]

def analyze_inst(instructions):
    '''
    Create vocabulary + extract all instructions splitted and in lower case
    '''
    split_instructions = []
    word_list = []
    max_sequence_length = 0
    for inst in instructions:
        split_inst = inst.lower().split(' ')
        len_inst = len(split_inst)
        if len_inst > max_sequence_length:
            max_sequence_length = len_inst
        word_list.extend(split_inst)
        split_instructions.append(split_inst)

    word_set = set(word_list)

    return split_instructions, max_sequence_length, word_set




class Vocab(object):
    '''
    Vocabulary class:
    id2word: mapping of index to word
    word2id mapping of words to index
    '''

    def __init__(self, words):
        word_list = sorted(list(set(words)))
        self.id2word = dict(zip([0] + [i + 1 for i in range(len(word_list))], ['pad'] + word_list))
        self.size = len(word_list) + 1  # +1 to account for padding
        self.word2id = dict(zip(['pad'] + word_list, [0] + [i + 1 for i in range(len(word_list))]))


class AbstractEncoder(ABC):
    '''
    Encoder must implement function encode and decode and be init with vocab and max_seq_length
    '''

    def __init__(self, vocab, max_seq_length):
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        super().__init__()

    @abstractmethod
    def encode(self, instruction):
        pass

    @abstractmethod
    def decode(self, sequence):
        pass


class OneHotEncoder(AbstractEncoder):

    def _word2one_hot(self, word):
        id = self.vocab.word2id[word]
        out = np.zeros(self.vocab.size)
        out[id] = 1
        return out

    def encode(self, split_instruction):
        one_hot_seq = []
        for word in split_instruction:
            one_hot_seq.append(self._word2one_hot(word))
        while len(one_hot_seq) < self.max_seq_length:
            one_hot_seq.append(np.zeros(self.vocab.size))
        return one_hot_seq

    def decode(self, one_hot_seq):
        words = []
        for vect in one_hot_seq:
            if np.sum(vect) > 0:
                words.append(self.vocab.id2word[np.where(vect > 0)[0][0]])
        return ' '.join(words)
