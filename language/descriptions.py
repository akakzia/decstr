from language.get_data import get_data
import numpy as np


def get_description(config_init, config_final, exhaustive=True):
    unique_reached_config_transitions, reached_config_transitions, predicates, \
    predicate_to_id, id_to_predicate, colors = get_data()

    diff = config_final - config_init
    sentences = []
    for i_d, d in enumerate(diff):
        p = predicates[i_d]
        if d == 1:
            positive = True
        elif d == -1:
            positive = False
        else:
            break

        words = p.split('_')
        for i in range(len(words)):
            try:
                words[i] = colors[words[i]]
            except:
                pass

        valid_sentences = []
        if words[0] == 'close':
            if positive:
                valid_sentences.append('Put {} close to {}'.format(words[1], words[2]))
                valid_sentences.append('Get {} close to {}'.format(words[1], words[2]))
                valid_sentences.append('Put {} close to {}'.format(words[2], words[1]))
                valid_sentences.append('Get {} close to {}'.format(words[2], words[1]))
                valid_sentences.append('Get {} and {} close from each other'.format(words[1], words[2]))
                valid_sentences.append('Get {} and {} close from each other'.format(words[2], words[1]))
                valid_sentences.append('Bring {} and {} together'.format(words[1], words[2]))
                valid_sentences.append('Bring {} and {} together'.format(words[2], words[1]))
            else:
                valid_sentences.append('Put {} far from {}'.format(words[1], words[2]))
                valid_sentences.append('Get {} far from {}'.format(words[1], words[2]))
                valid_sentences.append('Put {} far from {}'.format(words[2], words[1]))
                valid_sentences.append('Get {} far from {}'.format(words[2], words[1]))
                valid_sentences.append('Get {} and {} far from each other'.format(words[1], words[2]))
                valid_sentences.append('Get {} and {} far from each other'.format(words[2], words[1]))
                valid_sentences.append('Bring {} and {} apart'.format(words[1], words[2]))
                valid_sentences.append('Bring {} and {} apart'.format(words[2], words[1]))
        elif words[0] == 'above':
            if positive:
                valid_sentences.append('Put {} above {}'.format(words[1], words[2]))
                valid_sentences.append('Put {} on top of {}'.format(words[1], words[2]))
                valid_sentences.append('Put {} under {}'.format(words[2], words[1]))
                valid_sentences.append('Put {} below {}'.format(words[2], words[1]))
            else:
                valid_sentences.append('Put {} above {}'.format(words[2], words[1]))
                valid_sentences.append('Put {} on top of {}'.format(words[2], words[1]))
                valid_sentences.append('Put {} under {}'.format(words[1], words[2]))
                valid_sentences.append('Put {} below {}'.format(words[1], words[2]))
        else:
            raise NotImplementedError

        sentences += valid_sentences

    if not exhaustive:
        sentences = [np.random.choice(sentences)]

    return sentences
