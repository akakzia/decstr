import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
from language.build_dataset import sentence_from_configuration
from utils import get_instruction2

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

if __name__ == '__main__':
    num_eval = 1
    path = '/home/ahakakzia/language_baseline/1/'
    model_path = path + 'model_530.pt'

    with open(path + 'config.json', 'r') as f:
        params = json.load(f)
    args = SimpleNamespace(**params)

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    if args.agent == "SAC":
        policy = RLAgent(args, env.compute_reward, goal_sampler)
        policy.load(model_path, args)
    else:
        raise NotImplementedError

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    eval_goals = goal_sampler.valid_goals
    if args.algo == 'language':
        language_goal = get_instruction2()
        eval_goals = np.array([goal_sampler.valid_goals[0] for _ in range(len(language_goal))])
    else:
        language_goal = None
    inits = [None] * len(eval_goals)
    all_results = []
    for i in range(num_eval):
        episodes = rollout_worker.generate_rollout(eval_goals, self_eval=True, true_eval=True, animated=True, language_goal=language_goal)
        if args.algo == 'language':
            results = np.array([e['language_goal'] in sentence_from_configuration(e['ag'][-1], all=True) for e in episodes]).astype(np.int)
        elif args.algo == 'continuous':
            results = np.array([e['rewards'][-1] == 3. for e in episodes])
        else:
            results = np.array([str(e['g'][0]) == str(e['ag'][-1]) for e in episodes]).astype(np.int)
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))