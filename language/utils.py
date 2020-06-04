import numpy as np
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch

class ConfigLanguageDataset(Dataset):
    def __init__(self, configs, sentences, continuous,  string_to_1hot, binary=True, shuffle=True):

        assert configs.shape[0] == len(sentences)

        one_hots = []
        for s in sentences:
            one_hots.append(string_to_1hot[s.lower()])

        inds = np.arange(configs.shape[0])
        np.random.shuffle(inds)

        self.sentences = np.array(one_hots)[inds].astype(np.float32)
        self.configs = configs[inds].astype(np.float32)
        if continuous is not None:
            self.binary = False
            sents = self.sentences.copy()
            self.sentences = [np.expand_dims(s, 0) for s in sents]
            self.continuous = continuous[inds].astype(np.float32)
            #
            # unique, idx = np.unique(self.continuous, axis=0, return_inverse=True)
            # conts = []
            # configs = []
            # sentences = []
            # for id in range(unique.shape[0]):
            #     ids_in_array = np.argwhere(idx == id).flatten()
            #     conts.append(unique[id].copy())
            #     configs.append(self.configs[ids_in_array[0]].copy())
            #     sentences.append(self.sentences[ids_in_array].copy())
            # self.continuous = np.array(conts).copy()
            # self.configs = np.array(configs).copy()
            # self.sentences = sentences.copy()
        else:
            self.binary = True
            unique, idx = np.unique(self.configs, axis=0, return_inverse=True)
            configs = []
            sentences = []
            for id in range(unique.shape[0]):
                ids_in_array = np.argwhere(idx == id).flatten()
                configs.append(unique[id].copy())
                sentences.append(self.sentences[ids_in_array].copy())
            self.configs = np.array(configs).copy()
            self.sentences = sentences.copy()
        self.idx = []
        for i in range(self.configs.shape[0]):
            for j in range(self.sentences[i].shape[0]):
                self.idx.append([i, j])
        self.idx = np.array(self.idx)
        configs = None
        sentences = None
        continuous = None

    def __getitem__(self, index):
        if self.binary:
            idx = self.idx[index]
            return self.configs[idx[0]][0], self.sentences[idx[0]][idx[1]], self.configs[idx[0]][1]
        else:
            idx = self.idx[index]
            return self.configs[idx[0]][0], self.sentences[idx[0]][idx[1]], self.configs[idx[0]][1], self.continuous[idx[0]][0], self.continuous[idx[0]][1]


    def __len__(self):
        return self.idx.shape[0]

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


def generate_goals(nb_objects=3, sym=1, asym=1):
    """
    generates all the possible goal configurations whether feasible or not, then regroup them into buckets
    :return:
    """
    buckets = {0: [(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                   (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                1: [(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                2: [(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                3: [(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                     (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)],

                4:  [(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
                     (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                     ]}
    return buckets


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


def check_same_plane(config, color1, color2):
    predicates = ['close_0_1',
                  'close_0_2',
                  'close_1_2',
                  'above_0_1',
                  'above_1_0',
                  'above_0_2',
                  'above_2_0',
                  'above_1_2',
                  'above_2_1']

    colors = {'red':'0', 'green':'1', 'blue':'2'}
    id1 = int(colors[color1])
    id2 = int(colors[color2])
    above1 = bool(config[predicates.index('above_{}_{}'.format(id1, id2))])
    above2 = bool(config[predicates.index('above_{}_{}'.format(id2, id1))])
    if above1 or above2:
        return False
    for idx, p in enumerate(predicates[3:]):
        if 'above_{}'.format(id1) in p or 'above_{}'.format(id2) in p:
            if config[idx + 3] == 1:
                return False
    return True

def get_corresponding_sentences(config_inital, config_final):
    sentences = []

    predicates = ['close_0_1',
                  'close_0_2',
                  'close_1_2',
                  'above_0_1',
                  'above_1_0',
                  'above_0_2',
                  'above_2_0',
                  'above_1_2',
                  'above_2_1']

    colors = {'0':'red', '1':'green', '2':'blue'}

    # delta = config_final - config_init
    # # delta[np.logical_and(config_final, config_init)] = 1
    # delta[np.logical_and(config_final, config_init)] = 1


    for i, d in enumerate(config_final):
        p = predicates[i]
        words = p.split('_')
        for j in range(len(words)):
            try:
                words[j] = colors[words[j]]
            except:
                pass

        if words[0] == 'close':
            if d == 1:
                sentences.append('put {} close_to {}'.format(words[1], words[2]))
                sentences.append('get {} close_to {}'.format(words[1], words[2]))
                sentences.append('put {} close_to {}'.format(words[2], words[1]))
                sentences.append('get {} close_to {}'.format(words[2], words[1]))
                sentences.append('get {} and {} close_from each_other'.format(words[1], words[2]))
                sentences.append('get {} and {} close_from each_other'.format(words[2], words[1]))
                sentences.append('bring {} and {} together'.format(words[1], words[2]))
                sentences.append('bring {} and {} together'.format(words[2], words[1]))
            elif d == 0:
                sentences.append('put {} far_from {}'.format(words[1], words[2]))
                sentences.append('get {} far_from {}'.format(words[1], words[2]))
                sentences.append('put {} far_from {}'.format(words[2], words[1]))
                sentences.append('get {} far_from {}'.format(words[2], words[1]))
                sentences.append('get {} and {} far_from each_other'.format(words[1], words[2]))
                sentences.append('get {} and {} far_from each_other'.format(words[2], words[1]))
                sentences.append('bring {} and {} apart'.format(words[1], words[2]))
                sentences.append('bring {} and {} apart'.format(words[2], words[1]))
        elif words[0] == 'above':
            if d == 1:
                sentences.append('put {} above {}'.format(words[1], words[2]))
                sentences.append('put {} on_top_of {}'.format(words[1], words[2]))
                sentences.append('put {} under {}'.format(words[2], words[1]))
                sentences.append('put {} below {}'.format(words[2], words[1]))
            elif d == 0:
                sentences.append('remove {} from {}'.format(words[1], words[2]))
                sentences.append('remove {} from_above {}'.format(words[1], words[2]))
                sentences.append('remove {} from_under {}'.format(words[2], words[1]))
                sentences.append('remove {} from_below {}'.format(words[2], words[1]))

        else:
            raise NotImplementedError
    colors = list(colors.values())
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = colors[i], colors[j]
            if check_same_plane(config_final, a, b):
                sentences.append('put {} and {} on_the_same_plane'.format(a, b))
                sentences.append('put {} and {} on_the_same_plane'.format(b, a))
    for i in range(len(sentences)):
        sentences[i] = sentences[i].lower()
    return sentences.copy()




def get_list_of_expressions():
    expressions = []

    above_pos = []
    above_pos.append(('put {} above {}', True))
    above_pos.append(('put {} on_top_of {}', True))
    above_pos.append(('put {} under {}', False))
    above_pos.append(('put {} below {}', False))
    above_neg = []
    above_neg.append(('remove {} from {}', True))
    above_neg.append(('remove {} from_above {}', True))
    above_neg.append(('remove {} from_under {}', False))
    above_neg.append(('remove {} from_below {}', False))
    above_neg.append('put {} and {} on_the_same_plane')
    above_neg.append('put {} and {} on_the_same_plane')
    close_pos = []
    close_pos.append('put {} close_to {}')
    close_pos.append('get {} close_to {}')
    close_pos.append('put {} close_to {}')
    close_pos.append('get {} close_to {}')
    close_pos.append('get {} and {} close_from each_other')
    close_pos.append('get {} and {} close_from each_other')
    close_pos.append('bring {} and {} together')
    close_pos.append('bring {} and {} together')
    close_neg = []
    close_neg.append('put {} far_from {}')
    close_neg.append('get {} far_from {}')
    close_neg.append('put {} far_from {}')
    close_neg.append('get {} far_from {}')
    close_neg.append('get {} and {} far_from each_other')
    close_neg.append('get {} and {} far_from each_other')
    close_neg.append('bring {} and {} apart')
    close_neg.append('bring {} and {} apart')

    colors = ['red', 'green', 'blue']
    all_couples = []
    all_triplets = []
    for i in colors:
        for j in colors:
            if i != j:
                all_couples.append((i, j))
            for k in colors:
                if i!= j and j!=k and i!=k:
                    all_triplets.append((i, j, k))


    for _ in range(10):
        triplet = all_triplets[np.random.choice(range(len(all_triplets)))]
        ids = np.random.choice(range(len(above_pos)), size=2, replace=False)
        above_a = above_pos[ids[0]]
        above_b = above_pos[ids[1]]
        exp = np.random.choice(['pyramid', 'stack3'])
        if isinstance(above_a, tuple):
            if above_a[1]:
                first = above_a[0].format(triplet[0], triplet[1])
            else:
                first = above_a[0].format(triplet[1], triplet[0])
        else:
            first = above_a.format(triplet[0], triplet[1])

        if isinstance(above_b, tuple):
            if above_b[1]:
                if exp == 'pyramid':
                    second = above_b[0].format(triplet[0], triplet[2])
                elif exp == 'stack3':
                    second = above_b[0].format(triplet[1], triplet[2])
            else:
                if exp == 'pyramid':
                    second = above_b[0].format(triplet[2], triplet[0])
                elif exp == 'stack3':
                    second = above_b[0].format(triplet[2], triplet[1])
        else:
            if exp == 'pyramid':
                second = above_b.format(triplet[0], triplet[2])
            elif exp == 'stack3':
                second = above_b.format(triplet[1], triplet[2])
        expressions.append(['and', first, second])


    # one stack and third close or far OR another
    for _ in range(20):
        triplet = all_triplets[np.random.choice(range(len(all_triplets)))]
        stack = np.random.choice(above_pos + above_neg)
        close = np.random.choice(close_pos + close_neg)
        if isinstance(stack, tuple):
            stack = stack[0]

        pair = [triplet[2], np.random.choice(list(triplet[:2]))]
        np.random.shuffle(pair)
        if np.random.rand() < 0.8:
            exp1 = ['and', stack.format(triplet[0], triplet[1]), close.format(pair[0], pair[1])]
        else:
            exp1 = ['and', ['not',  stack.format(triplet[0], triplet[1])], close.format(pair[0], pair[1])]

        triplet = all_triplets[np.random.choice(range(len(all_triplets)))]
        stack = np.random.choice(above_pos + above_neg)
        close = np.random.choice(close_pos + close_neg)
        if isinstance(stack, tuple):
            stack = stack[0]

        pair = [triplet[2], np.random.choice(list(triplet[:2]))]
        np.random.shuffle(pair)
        if np.random.rand() < 0.8:
            exp2 = ['and', stack.format(triplet[0], triplet[1]), close.format(pair[0], pair[1])]
        else:
            exp2 = ['and', stack.format(triplet[0], triplet[1]), ['not', close.format(pair[0], pair[1])]]

        expressions.append(['or', exp1, exp2])

    # one stack and third close
    for _ in range(20):
        triplet = all_triplets[np.random.choice(range(len(all_triplets)))]
        stack = np.random.choice(above_pos + above_neg)
        close = np.random.choice(close_pos + close_neg)
        if isinstance(stack, tuple):
            stack = stack[0]

        pair = [triplet[2], np.random.choice(list(triplet[:2]))]
        np.random.shuffle(pair)
        if np.random.rand() < 0.8:
            exp1 = ['and', stack.format(triplet[0], triplet[1]), close.format(pair[0], pair[1])]
        else:
            if np.random.rand() < 0.5:
                exp1 = ['and', ['not', stack.format(triplet[0], triplet[1])], close.format(pair[0], pair[1])]
            else:
                exp1 = ['and', stack.format(triplet[0], triplet[1]), ['not', close.format(pair[0], pair[1])]]
        expressions.append(exp1)


    return expressions

def generate_all_goals_in_goal_space():
    goals = []
    for a in [0, 1]:
        for b in [0, 1]:
            for c in [0, 1]:
                for d in [0, 1]:
                    for e in [0, 1]:
                        for f in [0, 1]:
                            for g in [0, 1]:
                                for h in [0, 1]:
                                    for i in [0, 1]:
                                        goals.append([a, b, c, d, e, f, g, h, i])

    return np.array(goals)