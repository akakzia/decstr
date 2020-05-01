from language.get_data import get_data
import numpy as np


def get_dataset():
    unique_reached_config_transitions, reached_config_transitions, predicates, \
    predicate_to_id, id_to_predicate, colors = get_data()

    # construct dataset language
    data_configs =  []
    data_sentences = []
    for i in range(len(predicates)):
        p = predicates[i]
        words = p.split('_')
        for j in range(len(words)):
            try:
                words[j] = colors[words[j]]
            except:
                pass

        for positive in [True, False]:
            sentences = []
            if words[0] == 'close':
                if positive:
                    sentences.append('Put {} close_to {}'.format(words[1], words[2]))
                    sentences.append('Get {} close_to {}'.format(words[1], words[2]))
                    sentences.append('Put {} close_to {}'.format(words[2], words[1]))
                    sentences.append('Get {} close_to {}'.format(words[2], words[1]))
                    sentences.append('Get {} and {} close_from each_other'.format(words[1], words[2]))
                    sentences.append('Get {} and {} close_from each_other'.format(words[2], words[1]))
                    sentences.append('Bring {} and {} together'.format(words[1], words[2]))
                    sentences.append('Bring {} and {} together'.format(words[2], words[1]))
                else:
                    sentences.append('Put {} far_from {}'.format(words[1], words[2]))
                    sentences.append('Get {} far_from {}'.format(words[1], words[2]))
                    sentences.append('Put {} far_from {}'.format(words[2], words[1]))
                    sentences.append('Get {} far_from {}'.format(words[2], words[1]))
                    sentences.append('Get {} and {} far_from each_other'.format(words[1], words[2]))
                    sentences.append('Get {} and {} far_from each_other'.format(words[2], words[1]))
                    sentences.append('Bring {} and {} apart'.format(words[1], words[2]))
                    sentences.append('Bring {} and {} apart'.format(words[2], words[1]))
            elif words[0] == 'above':
                if positive:
                    sentences.append('Put {} above {}'.format(words[1], words[2]))
                    sentences.append('Put {} on_top_of {}'.format(words[1], words[2]))
                    sentences.append('Put {} under {}'.format(words[2], words[1]))
                    sentences.append('Put {} below {}'.format(words[2], words[1]))
                else:
                    sentences.append('Remove {} from {}'.format(words[1], words[2]))
                    sentences.append('Remove {} from_above {}'.format(words[1], words[2]))
                    sentences.append('Remove {} from_under {}'.format(words[2], words[1]))
                    sentences.append('Remove {} from_below {}'.format(words[2], words[1]))
                    sentences.append('Put {} and {} on_the_same_plane'.format(words[1], words[2]))
                    sentences.append('Put {} and {} on_the_same_plane'.format(words[2], words[1]))
            else:
                raise NotImplementedError
            for transition in unique_reached_config_transitions:
                if positive and transition[1][i] == transition[0][i] + 1:
                    data_configs += [transition.copy()] * len(sentences)
                    data_sentences += sentences
                elif not positive and transition[1][i] == transition[0][i] - 1:
                    data_configs += [transition.copy()] * len(sentences)
                    data_sentences += sentences

    data_configs = np.array(data_configs)
    sentences_set = set(data_sentences)
    # print(len(sentences_set))
    # for i in sentences_set:
    #     print(i)
    stop = 1

    return data_configs, data_sentences