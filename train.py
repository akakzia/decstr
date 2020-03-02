import numpy as np
import gym
import gym_object_manipulation
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
from rl_modules.sac_agent import SACAgent
import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""


def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params


def launch(args):
    if args.multi_criteria_her:
        assert args.env_name == 'FetchManipulate3ObjectsIncremental-v0'
    # create the ddpg_agent
    env = gym.make(args.env_name)
    # set random seeds for reproduce
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    process = '{}'.format(args.folder_prefix)
    if args.curriculum_learning:
        process = 'Curriculum_{}'.format(args.curriculum_eps)
    # Create save directory
    """if MPI.COMM_WORLD.Get_rank() == 0:
        if not os.path.exists(args.save_dir):
            os.mkdir(os.path.join(args.save_dir))
        if not os.path.exists(os.path.join(args.save_dir, '{}_{}'.format(args.env_name, process))):
            os.mkdir(os.path.join(args.save_dir, '{}_{}'.format(args.env_name, process)))"""
    # create the ddpg agent to interact with the environment
    if args.agent == "DDPG":
        ddpg_trainer = ddpg_agent(args, env, env_params)
        ddpg_trainer.learn()
    elif args.agent == "SAC":
        sac_trainer = SACAgent(args, env, env_params)
        sac_trainer.learn()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # take the configuration for the HER
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the params
    args = get_args()
    launch(args)
