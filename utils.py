import numpy as np
from datetime import datetime
import itertools
import os
import json


# def rollout(agent, animated=False):
#     ep_obs, ep_ag, ep_g, ep_actions, ep_success = [], [], [], [], []
#     eval = False
#     nb_buckets = agent.num_buckets
#     bucket = 0
#     #  Desired goal selection according to params
#     if agent.args.curriculum_learning:
#         if agent.args.automatic_buckets and ((np.array([len(bucket) < 1 for bucket in agent.buckets.values()])).any() or np.random.uniform() < 0.2):
#             # Randomly select a goal among all valid and non valid goals
#             goal = agent.goals[np.random.choice(np.arange(len(agent.goals)))]
#         else:
#             eval = True if np.random.random() < 0.1 else False
#             # select goal according to LP probability
#             if eval:
#                 bucket = np.random.choice(np.arange(nb_buckets))
#                 goal = agent.buckets[bucket][np.random.choice(len(agent.buckets[bucket]))]
#             else:
#                 bucket = np.random.choice(np.arange(nb_buckets), 1, p=agent.p)[0]
#                 goal = agent.buckets[bucket][np.random.choice(len(agent.buckets[bucket]))]
#     else:
#         flatten = lambda l: [item for sublist in list(agent.buckets.values()) for item in sublist]
#         goal = random.choice(flatten(agent.buckets))
#     observation = agent.env.unwrapped.reset_goal(np.array(goal), biased_init=agent.args.biased_init, eval=eval)
#     obs = observation['observation']
#     ag = observation['achieved_goal']
#     g = observation['desired_goal']
#     # start to collect samples
#     for t in range(agent.env_params['max_timesteps']):
#         # if an ag is not encountred by all workers, add it
#         if tuple(ag) not in sum(MPI.COMM_WORLD.allgather(agent.encountred_goals), []):
#             agent.encountred_goals.append(tuple(ag))
#             # if the encountred goal is new and was not predefined as a valid_goal, then add it
#             if tuple(ag) not in agent.valid_goals:
#                 agent.valid_goals.append(tuple(ag))
#                 agent.per_goal_competence_computers.append(CompetenceQueue(window=agent.args.queue_length))
#         with torch.no_grad():
#             obs_norm = agent.o_norm.normalize(obs)
#             g_norm = torch.tensor(agent.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
#             ag_norm = torch.tensor(agent.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)
#             if agent.architecture == 'deepsets':
#                 obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
#                 agent.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, eval=eval)
#                 action = agent.model.pi_tensor.numpy()[0]
#             elif agent.architecture == 'disentangled':
#                 z_ag = agent.configuration_network(ag_norm)[0]
#                 z_g = agent.configuration_network(g_norm)[0]
#                 input_tensor = torch.tensor(np.concatenate([obs_norm, z_ag, z_g]), dtype=torch.float32).unsqueeze(0)
#                 action = agent._select_actions(input_tensor, eval=eval)
#             else:
#                 input_tensor = agent._preproc_inputs(obs, g)  # PROCESSING TO CHECK
#                 action = agent._select_actions(input_tensor, eval=eval)
#         # feed the actions into the environment
#         if animated:
#             agent.env.render()
#         observation_new, _, _, info = agent.env.step(action)
#         obs_new = observation_new['observation']
#         ag_new = observation_new['achieved_goal']
#         ep_success = info['is_success']  #
#         # append rollouts
#         ep_obs.append(obs.copy())
#         ep_ag.append(ag.copy())
#         ep_g.append(g.copy())
#         ep_actions.append(action.copy())
#         # re-assign the observation
#         obs = obs_new
#         ag = ag_new
#     ep_obs.append(obs.copy())
#     ep_ag.append(ag.copy())
#     return ep_obs, ep_ag, ep_g, ep_actions, ep_success, eval, g, bucket



def above_to_close(vector):
    """
    Given a configuration of above objects, determines a configuration of close objects
    :param vector:
    :return:
    """
    size = len(vector)
    res = np.zeros(size//2)
    for i in range(size//2):
        if vector[2*i] == 1. or vector[2*i+1] == 1.:
            res[i] = 1.
    return tuple(res)


def valid(vector):
    """
    Determines whether an above configuration is valid or not
    :param vector:
    :return:
    """
    size = len(vector)
    if sum(vector) > 2:
        return False
    else:
        """can't have x on y and y on x"""
        for i in range(size//2):
            if vector[2*i] == 1. and vector[2*i] == vector[2*i+1]:
                return False
        """can't have two blocks on one blocks"""
        if (vector[0] == 1. and vector[0] == vector[-1]) or \
                (vector[1] == 1. and vector[1] == vector[3]) or (vector[2] == 1. and vector[2] == vector[4]):
            return False
    return True


def one_above_two(vector):
    """
    Determines whether one block is above two blocks
    """
    if (vector[0] == 1. and vector[0] == vector[2]) or \
            (vector[1] == 1. and vector[1] == vector[-2]) or (vector[3] == 1. and vector[3] == vector[-1]):
        return True
    return False


stack_three_list = [(1., 1., 0., 1., 0., 0., 1., 0., 0.), (1., 0., 1., 0., 1., 0., 0., 0., 1.),
                    (1., 1., 0., 0., 1., 1., 0., 0., 0.), (1., 0., 1., 1., 0., 0., 0., 1., 0.),
                    (0., 1., 1., 0., 0., 1., 0., 0., 1.), (0., 1., 1., 0., 0., 0., 1., 1., 0.)]


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
                    (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
                3: [(1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                     (1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                4: [(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)],

                5:  [(1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                     (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0)],

                6:  [(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
                     (1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                     (1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
                     (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0), (0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0),
                     (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                     (1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                     ]}
    return buckets
    # size = sym * nb_objects * (nb_objects - 1) // 2 + asym * nb_objects * (nb_objects - 1)
    # all_configurations = itertools.product([0., 1.], repeat=size)
    # """if asym < 1:
    #     for configuration in all_configurations:
    #         if sum(configuration) == 0. or (sum(configuration[:3]) == 1. and sum(configuration[-3:]) == 0.):
    #             # All far and only one pair is close
    #             buckets[0].append(configuration)
    #         elif sum(configuration[:3]) > 1 and sum(configuration[-3:]) == 0.:
    #             # Two oor three pairs are close
    #             buckets[1].append(configuration)
    #         elif (configuration[:3] == configuration[-3:] and sum(configuration) == 2.) or \
    #                 (sum(configuration[:3]) == 3. and sum(configuration[-3:]) == 2.):
    #             # Only stacks are close
    #             buckets[2].append(configuration)
    #         elif (np.array(configuration[:3]) - np.array(configuration[-3:]) != -1.).all():
    #             # Other configurations
    #             buckets[3].append(configuration)
    #         else:
    #             # Not feasible
    #             buckets[4].append(configuration)
    # else:
    #     for configuration in all_configurations:
    #         if not valid(configuration[-6:]):
    #             # Not feasible having more than 5 ones | Not feasible that two blocks are above a single block
    #             buckets[5].append(configuration)
    #         elif sum(configuration[:3]) <= 1. and sum(configuration[-6:]) == 0.:
    #             # All far and only one pair is close
    #             buckets[0].append(configuration)
    #         elif sum(configuration[:3]) > 1 and sum(configuration[-6:]) == 0.:
    #             # Two or three pairs are close
    #             buckets[1].append(configuration)
    #         elif min(np.array(configuration[:3]) - np.array(above_to_close(configuration[-6:]))) == 0.0 and sum(configuration[-6:]) == 1.:
    #             # Only stacks are close
    #             buckets[2].append(configuration)
    #         elif one_above_two(configuration[-6:]) and sum(configuration) == 5.:
    #             # Other configurations
    #             buckets[3].append(configuration)
    #         elif min(np.array(configuration[:3]) - np.array(above_to_close(configuration[-6:]))) == 0.0 and \
    #                 not one_above_two(configuration[-6:]) and sum(configuration) == 4.:
    #             # Other configurations
    #             buckets[4].append(configuration)
    #         else:
    #             # Not feasible
    #             buckets[5].append(configuration)"""
    # if asym < 1:
    #     buckets = {0: [], 1: [], 2: []}
    #     for configuration in all_configurations:
    #         if sum(configuration) < 2.:
    #             buckets[0].append(configuration)
    #         elif sum(configuration) == 2.:
    #             buckets[1].append(configuration)
    #         else:
    #             buckets[2].append(configuration)
    # else:
    #     buckets = {0: [], 1: [], 2: [], 3: [], 4: []}
    #     for configuration in all_configurations:
    #         if sum(configuration) == 0. or (sum(configuration[:3]) == 1. and sum(configuration[-6:]) == 0.):
    #             # All far and only one pair is close
    #             buckets[0].append(configuration)
    #         elif sum(configuration[:3]) > 1 and sum(configuration[-6:]) == 0.:
    #             # Two or three pairs are close
    #             buckets[1].append(configuration)
    #         elif configuration[:3] == above_to_close(configuration[-6:]) and sum(configuration) == 2.:
    #             # Only stacks are close
    #             buckets[2].append(configuration)
    #         elif sum(configuration[:3]) == 3. and valid(configuration[-6:]):
    #             if sum(configuration[-6:]) == 1:
    #                 # Stack of 2 close to the third block
    #                 buckets[3].append(configuration)
    #             elif configuration[-6:] in [(1., 0., 1., 0., 0., 0.), (0., 1., 0., 0., 1., 0.), (0., 0., 0., 1., 0., 1.)]:
    #                 # One block above two blocks (pyramid)
    #                 buckets[3].append(configuration)
    #         elif configuration in stack_three_list:
    #             # Stack of 3
    #             buckets[3].append(configuration)
    #         else:
    #             # Not feasible
    #             buckets[4].append(configuration)
    # return buckets


def init_storage(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # path to save the model
    logdir = os.path.join(args.save_dir, args.env_name + '_' + args.folder_prefix)
    if args.curriculum_learning:
        logdir = os.path.join(args.save_dir, '{}_curriculum_{}'.format(datetime.now(), args.architecture))
        if args.deepsets_attention:
            logdir += '_attention'
        if args.double_critic_attention:
            logdir += '_double'
    # path to save evaluations
    model_path = os.path.join(logdir, 'models')
    bucket_path = os.path.join(logdir, 'buckets')
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(bucket_path):
        os.mkdir(bucket_path)
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return logdir, model_path, bucket_path
