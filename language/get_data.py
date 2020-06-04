import pickle


def get_data(binary=True):

    path_config_transitions_reached = "./data/learned_configs_continuous.pkl"

    with open(path_config_transitions_reached, 'rb') as f:
        reached_config_transitions = pickle.load(f)


    reached_config_transitions = reached_config_transitions[:6000]

    predicates = ['close_0_1',
                  'close_0_2',
                  'close_1_2',
                  'above_0_1',
                  'above_1_0',
                  'above_0_2',
                  'above_2_0',
                  'above_1_2',
                  'above_2_1']


    predicate_to_id = dict(zip(predicates, range(9)))
    id_to_predicate = dict(zip(range(9), predicates))

    colors = {'0':'red', '1':'green', '2':'blue'}

    return reached_config_transitions,  predicates, predicate_to_id, id_to_predicate, colors