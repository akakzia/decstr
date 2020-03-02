import numpy as np
import torch
import itertools
import os
import json


def rollout(env, env_params, agent, args, goals, animated=False):
    """
    Description
    :param env:
    :param env_params:
    :param agent:
    :param args:
    :param goals:
    :param animated
    :return:
    """
    ep_obs, ep_ag, ep_g, ep_actions, ep_success = [], [], [], [], []
    eval = False
    nb_buckets = len(list(goals.keys()))
    bucket = 0
    #  Desired goal selection according to params
    if args.curriculum_learning:
        eval = True if np.random.random() < 0.1 else False  #
        # select goal according to LP probability
        if eval:
            #  goal = self.goals[np.random.choice(len(self.goals), 1)][0]
            #  goals_for_competence.append(goal)
            bucket = np.random.choice(np.arange(nb_buckets))
            goal = goals[bucket][np.random.choice(len(goals[bucket]))]
        else:
            #  goal = self.goals[np.random.choice(len(self.goals), 1, p=self.p)][0]
            bucket = np.random.choice(np.arange(nb_buckets), 1, p=agent.p)[0]
            goal = goals[bucket][np.random.choice(len(goals[bucket]))]
    else:
        # goal = goals[np.random.choice(len(goals), 1)][0]
        pass
    observation = env.reset_goal(np.array(goal))
    obs = observation['observation']
    ag = observation['achieved_goal']
    g = observation['desired_goal']
    # start to collect samples
    for t in range(env_params['max_timesteps']):
        with torch.no_grad():
            if agent.architecture == 'disentangled':
                g_norm = torch.tensor(agent.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)
                ag_norm = torch.tensor(agent.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)
                z_ag = agent.configuration_network(ag_norm)[0]
                z_g = agent.configuration_network(g_norm)[0]
                """input_tensor = torch.tensor(np.concatenate([agent.o_norm.normalize(obs),
                                                            z_ag, z_g]), dtype=torch.float32).unsqueeze(0)"""
                obs_tensor = torch.tensor(agent.o_norm.normalize(obs), dtype=torch.float32).unsqueeze(0)
                ag_tensor = z_ag.clone().detach().unsqueeze(0)
                g_tensor = z_g.clone().detach().unsqueeze(0)
                agent.model.forward_pass(obs_tensor, ag_norm, g_norm)
                action = agent.model.pi_tensor.numpy()

            else:
                input_tensor = agent._preproc_inputs(obs, g)  # PROCESSING TO CHECK
            #action = agent._select_actions(input_tensor, eval=eval)
        # feed the actions into the environment
        if animated:
            env.render()
        observation_new, _, _, info = env.step(action)
        obs_new = observation_new['observation']
        ag_new = observation_new['achieved_goal']
        ep_success = info['is_success']  #
        # append rollouts
        ep_obs.append(obs.copy())
        ep_ag.append(ag.copy())
        ep_g.append(g.copy())
        ep_actions.append(action.copy())
        # ep_success.append(success.copy())  #
        # re-assign the observation
        obs = obs_new
        ag = ag_new
    ep_obs.append(obs.copy())
    ep_ag.append(ag.copy())
    return ep_obs, ep_ag, ep_g, ep_actions, ep_success, eval, g, bucket


def load_models(path, actor, critic):
    o_mean, o_std, g_mean, g_std, model_actor, model_critic = torch.load(path, map_location=lambda storage, loc: storage)
    actor.load_state_dict(model_actor)
    critic.load_state_dict(model_critic)
    return o_mean, o_std, g_mean, g_std


def min_max_norm(vector):
    if vector.min() == vector.max():
        return np.ones(vector.shape[0])
    return (vector-vector.min()) / (vector.max() - vector.min())


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


def generate_goals(nb_objects=3, sym=1, asym=1):
    """
    generates all the possible goal configurations whether feasible or not, then regroup them into buckets
    :return:
    """
    buckets = {0: [], 1: [], 2: [], 3: [], 4: []}
    size = sym * nb_objects * (nb_objects - 1) // 2 + asym * nb_objects * (nb_objects - 1)
    all_configurations = itertools.product([0., 1.], repeat=size)
    """for configuration in all_configurations:
        if sum(configuration) == 0. or (sum(configuration[:3]) == 1. and sum(configuration[-3:]) == 0.):
            # All far and only one pair is close
            buckets[0].append(configuration)
        elif sum(configuration[:3]) > 1 and sum(configuration[-3:]) == 0.:
            # Two oor three pairs are close
            buckets[1].append(configuration)
        elif (configuration[:3] == configuration[-3:] and sum(configuration) == 2.) or \
                (sum(configuration[:3]) == 3. and sum(configuration[-3:]) == 2.):
            # Only stacks are close
            buckets[2].append(configuration)
        elif (np.array(configuration[:3]) - np.array(configuration[-3:]) != -1.).all():
            # Other configurations
            buckets[3].append(configuration)
        else:
            # Not feasible
            buckets[4].append(configuration)"""
    for configuration in all_configurations:
        if sum(configuration) == 0. or (sum(configuration[:3]) == 1. and sum(configuration[-6:]) == 0.):
            # All far and only one pair is close
            buckets[0].append(configuration)
        elif sum(configuration[:3]) > 1 and sum(configuration[-6:]) == 0.:
            # Two or three pairs are close
            buckets[1].append(configuration)
        elif configuration[:3] == above_to_close(configuration[-6:]) and sum(configuration) == 2.:
            # Only stacks are close
            buckets[2].append(configuration)
        elif sum(configuration[:3]) == 3. and valid(configuration[-6:]):
            # Other configurations
            buckets[3].append(configuration)
        else:
            # Not feasible
            buckets[4].append(configuration)
    return buckets


def init_storage(args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    # path to save the model
    model_path = os.path.join(args.save_dir, args.env_name + '_' + args.folder_prefix)
    if args.curriculum_learning:
        model_path = os.path.join(args.save_dir, '{}_Curriculum_{}'.format(args.env_name, args.folder_prefix))
    # path to save evaluations
    eval_path = os.path.join(model_path, 'eval')
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    return model_path, eval_path
